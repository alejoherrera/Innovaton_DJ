# rag_chain.py (Versión original adaptada para Cloud Run)
from __future__ import annotations
import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- Dependencias Clave ---
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from google.api_core.exceptions import ResourceExhausted
from langchain_community.document_loaders import PyPDFLoader # Usaremos LangChain para cargar el PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter # y para dividirlo

# Importamos las funciones para descargar desde Drive que ya creamos
from .drive_utils import download_file_from_drive, list_pdf_files_in_folder

# ── Configuración para Cloud Run ────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ID de la CARPETA de Google Drive (¡DEBES REEMPLAZAR ESTO!)
DRIVE_FOLDER_ID = "ID_DE_TU_CARPETA_DE_GOOGLE_DRIVE" 

if API_KEY:
    genai.configure(api_key=API_KEY)

# --- Variables Globales y de Estado ---
# Usaremos una variable global para saber si el sistema ya fue inicializado
IS_INITIALIZED = False
_LAST_ACTIVE: Dict[str, str] = {}  # memoria corta del acto activo

# --- Inicialización del Cliente ChromaDB y Modelos ---
# Se crea un cliente en memoria, no persistente
client = chromadb.Client() 
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY, model_name="models/embedding-001"
) if API_KEY else None
llm = genai.GenerativeModel(MODEL) if API_KEY else None
col = client.get_or_create_collection("resoluciones", embedding_function=emb_fn)

# ── Regex & Claves (Sin cambios) ────────────────────────────────
RES_RE     = re.compile(r"\b\d{4,6}-\d{4}\b")
INT_RE     = re.compile(r"\b[A-Z]{2}-\d{3,4}\b")
YEAR_RE    = re.compile(r"\b(19|20)\d{2}\b")
LIST_RE    = re.compile(r"\b(lista|listado|muestr[ae]|mostrar|dame|ens[eñ]a|ensena)\b", re.I)
HELLO_RE   = re.compile(r"\b(hola|buen[oa]s(?:\s*d[ií]as|\s*tardes|\s*noches)?|saludos|qu[eé] tal)\b", re.I)
GOODBYE_RE = re.compile(r"\b(ad[ií]os|hasta luego|nos vemos|chao|bye|hasta pronto)\b", re.I)
COURTESY_RE = re.compile(r"\b(gracias|muchas gracias|perfecto|de acuerdo|entendido)\b", re.I)
KEYWORDS_RAG = {"resolución","resolucion","acto final","expediente","número interno","numero interno","dj-","pa-","nn","número de resolución","numero de resolución","folio"}
SANCION_KEYS = {
    "despido sin responsabilidad": re.compile(r"despido\s+sin\s+responsabilidad", re.I),
    "despido con responsabilidad": re.compile(r"despido\s+con\s+responsabilidad", re.I),
    "suspensión": re.compile(r"suspensi[oó]n", re.I),
    "inhabilitación": re.compile(r"inhabilitaci[oó]n", re.I),
    "multa": re.compile(r"multa", re.I),
    "archivo": re.compile(r"archivo", re.I),
    "apercibimiento": re.compile(r"apercibimiento", re.I),
}

# --- NUEVA FUNCIÓN DE INICIALIZACIÓN ---
def initialize_rag_system():
    """
    Descarga los PDFs, los procesa y los carga en la base de datos ChromaDB en memoria.
    Esta función se ejecuta solo una vez al inicio.
    """
    global IS_INITIALIZED, col
    print("Iniciando sistema RAG...")
    
    # 1. Listar archivos PDF en la carpeta de Drive
    pdf_files = list_pdf_files_in_folder(DRIVE_FOLDER_ID)
    if not pdf_files:
        raise RuntimeError(f"No se encontraron archivos PDF en la carpeta de Drive con ID: {DRIVE_FOLDER_ID}")

    all_docs = []
    # 2. Descargar y procesar cada archivo PDF
    for pdf_info in pdf_files:
        pdf_id = pdf_info['id']
        pdf_name = pdf_info['name']
        print(f"Descargando y procesando: {pdf_name} (ID: {pdf_id})")
        
        local_pdf_path = download_file_from_drive(pdf_id, pdf_name)
        if not local_pdf_path:
            print(f"ADVERTENCIA: No se pudo descargar el archivo {pdf_name}")
            continue

        loader = PyPDFLoader(local_pdf_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs = text_splitter.split_documents(pages)
        all_docs.extend(docs)
        
        # Opcional: borrar el archivo descargado para ahorrar espacio en el contenedor
        os.remove(local_pdf_path)

    if not all_docs:
        raise RuntimeError("No se pudo procesar ningún documento PDF.")

    # 3. Preparar documentos para ChromaDB
    documents_to_add = [doc.page_content for doc in all_docs]
    # (Aquí necesitarías una lógica más avanzada para extraer metadatos de cada 'doc')
    metadatas_to_add = [{"source": doc.metadata.get("source", "N/A")} for doc in all_docs]
    ids_to_add = [f"doc_{i}" for i in range(len(documents_to_add))]
    
    # 4. Añadir los documentos a la colección en memoria
    print(f"Añadiendo {len(documents_to_add)} fragmentos a la base de datos en memoria...")
    col.add(
        documents=documents_to_add,
        metadatas=metadatas_to_add,
        ids=ids_to_add
    )
    
    IS_INITIALIZED = True
    print("¡Sistema RAG inicializado exitosamente!")

# ── Helpers (Tu código original) ────────────────────────────────
_norm = lambda s: " ".join((s or "").split())

def _sancion_tipo_simple(texto: str | None, tipo: str | None = None) -> str | None:
    if tipo and tipo.strip(): return tipo
    if not texto: return None
    low = texto.lower()
    for etiqueta, patt in SANCION_KEYS.items():
        if patt.search(low): return etiqueta
    return None

def _table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    if not rows: return "No se encontraron registros que cumplan con la condición solicitada."
    sep = "|".join(["---" for _ in headers])
    out = [" | ".join(headers), sep]
    out += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows]
    return "\n".join(out)

# ... (El resto de tus funciones helper: _list, _pick_idx, etc. irían aquí si las necesitas)
# Por simplicidad, las omitimos por ahora, ya que la lógica principal de 'answer' se centrará en la consulta a ChromaDB.

# ── Prompt ficha (Tu código original) ───────────────────────────
PROMPT_FICHA = (
    "Usted es **Lexi**, asistente virtual de la División Jurídica de la CGR (Costa Rica).\n"
    "Tono claro y profesional. Responda solo con datos presentes en el ‘Contexto’.\n\n"
    # ... (resto de tu prompt)
)

build_prompt = lambda **kw: PROMPT_FICHA.format(
    resol=kw.get("resol") or "desconocido",
    interno=kw.get("interno") or "desconocido",
    sancion=kw.get("sancion") or "sin indicios en metadatos",
    context=_norm(kw.get("context")) or "",
    query=_norm(kw.get("query")) or "",
)

# ── Generación robusta (Tu código original) ─────────────────────
def safe_generate(prompt: str, retries: int = 2) -> str:
    if not llm: return ""
    delay = 2.0
    for a in range(retries + 1):
        try:
            resp = llm.generate_content(prompt, generation_config={"temperature":0.2, "max_output_tokens":1024})
            txt = (getattr(resp, "text", "") or "").strip()
            if txt: return txt
        except ResourceExhausted:
            if a >= retries: return "⚠️ Se alcanzó la cuota de Gemini. Inténtelo más tarde."
        except Exception:
            pass
        time.sleep(delay); delay = min(delay*1.8, 10.0)
    return ""

# ── Modo libre (Tu código original) ─────────────────────────────
MSG_INICIAL  = (
    "¡Hola! 👋 Con mucho gusto le ayudo. Para buscar un acto final, indíqueme el "
    "**número de resolución** (p. ej. 07685-2025) o el **número interno** (p. ej. DJ-0612)."
)
MSG_DESPEDIDA = "¡Gracias por escribir! Si necesita otra consulta, aquí estaré. 👋"

# ── Router principal (MODIFICADO para la nube) ───────────────────
def answer(query: str, k: int = 10, debug: bool = False):
    """
    Función principal que procesa la consulta del usuario.
    Ahora se asegura de que el sistema esté inicializado antes de continuar.
    """
    global IS_INITIALIZED
    
    if not IS_INITIALIZED:
        try:
            initialize_rag_system()
        except Exception as e:
            print(f"ERROR FATAL DURANTE LA INICIALIZACIÓN: {e}")
            return "⚠️ Lo siento, el sistema no pudo iniciarse correctamente. Por favor, contacte al administrador.", []

    q = (query or "").strip()
    t = q.lower()

    # --- Lógica de conversación (simplificada de tu original) ---
    if GOODBYE_RE.search(t): return MSG_DESPEDIDA, []
    if HELLO_RE.search(t): return MSG_INICIAL, []
    if COURTESY_RE.search(t): return "¡Con mucho gusto! ¿Desea consultar alguna resolución o expediente?", []

    # --- Lógica RAG ---
    print(f"Realizando consulta a ChromaDB con: '{q}'")
    res = col.query(query_texts=[q], n_results=k, include=["documents", "metadatas"])
    
    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    
    if not docs:
        return "No se encontró información relevante en los documentos para su consulta.", []

    # Construimos un contexto simple con los resultados
    context = "\n\n---\n\n".join(docs)
    
    # (Aquí podrías re-integrar tu lógica más avanzada de _group_same_case, desambiguación, etc.)
    
    # Generamos la respuesta final
    prompt = build_prompt(context=context, query=q)
    final_response = safe_generate(prompt)
    
    if not final_response:
        final_response = "No pude generar una respuesta a partir de la información encontrada."

    return final_response, metas

# Export para app.py
__all__ = ["answer", "GOODBYE_RE"]
