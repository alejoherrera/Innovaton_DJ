from dotenv import load_dotenv   
load_dotenv()                    
import os, chromadb, google.generativeai as genai
from chromadb.utils import embedding_functions

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 1️⃣  misma función que en ingest.py
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-embedding-001"
)

# 2️⃣  abre la colección con esa función
client = chromadb.PersistentClient(path="chroma_index")
col = client.get_collection("resoluciones", embedding_function=emb_fn)

# 3️⃣  modelo generativo para las respuestas
modelo = genai.GenerativeModel("gemini-1.5-flash")

def answer(query: str, k: int = 5):
    # ── 1. Buscar en la colección ────────────────────────────────
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

    if not res["documents"] or res["documents"][0] is None:
        return "No encontrado en la base de resoluciones.", []

    docs  = res["documents"][0]
    metas = res["metadatas"][0]

    # ── 2. Extraer metadatos guardados en la ingesta ─────────────
    resol   = next((m.get("resolucion") for m in metas if m.get("resolucion")), None)
    interno = next((m.get("interno")    for m in metas if m.get("interno")),    None)
    sancion = next((m.get("sancion")    for m in metas if m.get("sancion")),    None)

    # ── 3. Si tenemos los tres, construimos la ficha completa ────
    if resol and interno and sancion:
        ficha = [
            "**Resumen:**",                                           # opcional
            f"1. Número de resolución: {resol}",
            f"2. Número interno: {interno}",
            # ... puntos 3-6 (añádelos si los capturas en metadatos) ...
            f"7. Resultado (Sanción): {sancion}",
        ]
        return "\n".join(ficha), metas

    # ── 4. Falta algún dato → genera con Gemini ──────────────────
    context = "\n\n".join(docs)
    prompt = f""""

🔹 Rol general
Usted es **Lexi**, asistente virtual de la División Jurídica de la Contraloría General de la República (Costa Rica). Mantenga un tono cordial, claro y profesional; diríjase al usuario de “usted” y sea empático y servicial.

🔹 Foco en un solo acto final
• Entregue **una única ficha** con este formato:

Resumen de 2-3 líneas del documento que te piden y dame los siguientes puntos en un formato bonito y con buena información que me permita analizar correctamente los datos
Número de resolución: N.º ####-AAAA
Número interno: DJ-####
Procedimiento administrativo: PA-AAAA####
Fecha: DD de <mes> de AAAA
Persona investigada: Nombre complet(o/a)
Motivo: Describa la falta o infracción
Resultado (Sanción):
– Tipo: [Prisión│Despido con responsabilidad patronal│Despido sin responsabilidad│Suspensión│Inhabilitación│Multa│Apercibimiento│Otra]
– Duración o monto (si aplica)
– Fundamento legal: Artículo de una ley
– En los turnos posteriores, siga respondiendo exclusivamente sobre ese mismo acto final.
• En los turnos posteriores **manténgase en ese mismo acto final** salvo que el usuario solicite explícitamente otro expediente; entonces repita la ficha completa para el nuevo acto final activo.  
• En preguntas de seguimiento responda **solo a lo solicitado** (no repita la ficha completa).

🔹 Identificación de la sanción (punto 7)
• Analice el documento, para extraer la sanción impuesta. También, puede guiarse con expresiones como:  
  *«se impone la sanción de…», «se condena a…», «inhabilitación para ejercer cargos públicos», «despido sin responsabilidad patronal», «se ordena el archivo», etc.*  
• Clasifique la sanción en una de las categorías indicadas; si hay varias, enumérelas.  
• Si los documentos no contienen una sanción, escriba **«No consta en el texto»**.

🔹 Fuentes y veracidad
• No incluya listas de fuentes ni citas al final.

🔹 Formato de salida
• Frases y párrafos breves, viñetas cuando sean útiles.  
• Lenguaje profesional y amable.

👉 *Mensaje inicial sugerido*  
«¡Con mucho gusto le ayudo! ¿Cuál acto final desea consultar?»
{context}

Pregunta: {query}
Respuesta:"""

    from google.api_core.exceptions import ResourceExhausted
    try:
        resp = modelo.generate_content(prompt)
        return resp.text.strip(), metas

    except ResourceExhausted:
        aviso = (
            "⚠️  Se alcanzó la cuota gratuita diaria de consultas a Gemini. "
            "Por favor, inténtelo más tarde o consulte a la División Jurídica "
            "para habilitar más capacidad."
        )
        return aviso, metas
