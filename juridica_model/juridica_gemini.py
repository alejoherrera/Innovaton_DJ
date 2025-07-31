import os
import json
from typing import Dict
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# === CONFIGURACIÓN ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelo = genai.GenerativeModel("gemini-1.5-flash")

INPUT_FOLDER = "G:/.shortcut-targets-by-id/1GPJzjEdw-lIyffYNZ4zFLI7pOCZfDk9z/Resoluciones"
OUTPUT_JSON = "pares_QA.json"

PREGUNTAS = {
    "numero_resolucion": "¿Cuál es el número de la resolución final?",
    "numero_interno": "¿Cuál es el número interno (DJ)?",
    "procedimiento_administrativo": "¿Cuál es el número del procedimiento administrativo?",
    "fecha_completa": "¿En qué fecha se emitió la resolución?",
    "persona_investigada": "¿Contra quién se siguió el procedimiento administrativo?",
    "motivo": "¿Cuál fue el motivo del procedimiento sancionador?",
    "resultado": "¿Cuál fue el resultado o la sanción impuesta?"
}

# === FUNCIONES PRINCIPALES ===

def extraer_texto_pdf(ruta_pdf: str) -> str:
    try:
        lector = PdfReader(ruta_pdf)
        texto = "\n".join([p.extract_text() for p in lector.pages if p.extract_text()])
        return texto
    except Exception as e:
        print(f"❌ Error leyendo {ruta_pdf}: {e}")
        return ""

def hacer_preguntas_lote(texto: str) -> Dict[str, str]:
    prompt = f"""Analiza el siguiente texto legal y responde brevemente las siguientes preguntas. 
Responde usando el siguiente formato JSON, usando exactamente las claves proporcionadas.

Texto:
\"\"\"{texto}\"\"\"

Preguntas:
{json.dumps(PREGUNTAS, indent=2, ensure_ascii=False)}

Formato esperado:
{{
  "numero_resolucion": "...",
  "numero_interno": "...",
  "procedimiento_administrativo": "...",
  "fecha_completa": "...",
  "persona_investigada": "...",
  "motivo": "...",
  "resultado": "..."
}}"""

    try:
        respuesta = modelo.generate_content(prompt)
        texto_respuesta = respuesta.text.strip()

        # Intentar extraer el bloque JSON
        json_start = texto_respuesta.find('{')
        json_end = texto_respuesta.rfind('}') + 1
        json_str = texto_respuesta[json_start:json_end]

        return json.loads(json_str)
    except Exception as e:
        print(f"❌ Error al preguntar a Gemini (lote): {e}")
        return {clave: "ERROR" for clave in PREGUNTAS.keys()}

def analizar_resoluciones():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ La carpeta '{INPUT_FOLDER}' no existe o no está accesible desde Python.")
        return

    archivos_pdf = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]
    if not archivos_pdf:
        print(f"⚠️ No se encontraron archivos PDF en la carpeta '{INPUT_FOLDER}'.")
        return

    print(f"📂 Archivos encontrados: {len(archivos_pdf)}")
    resultados = []

    for archivo in tqdm(archivos_pdf, desc="Analizando resoluciones"):
        ruta = os.path.join(INPUT_FOLDER, archivo)
        texto = extraer_texto_pdf(ruta)
        if not texto:
            continue

        respuestas = hacer_preguntas_lote(texto)
        resultados.append({
            "file_name": archivo,
            "respuestas": respuestas
        })

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"✅ Resultados guardados en: {OUTPUT_JSON}")

if __name__ == "__main__":
    analizar_resoluciones()
