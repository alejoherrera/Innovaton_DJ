# rag_chain.py (Versión original adaptada para Cloud Run con Qdrant)
from __future__ import annotations
import os
import re
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --- Dependencias Clave ---
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_community.document_loaders import PyPDFLoader # Usaremos LangChain para cargar el PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter # y para dividirlo

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Importamos las funciones para descargar desde Drive (import absoluto)
from drive_utils import download_file_from_drive, list_pdf_files_in_folder

# ── Configuración para Cloud Run ────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Configuración de Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "resoluciones"

# ID de la CARPETA de Google Drive
DRIVE_FOLDER_ID = "16as2spSPhK7027oqYer372k4Cxt_XOyf"

print("🚀 Iniciando rag_chain.py...")
print(f"Variables configuradas - GEMINI_API_KEY: {'OK' if API_KEY else 'FALTA'}")
print(f"QDRANT_URL: {'OK' if QDRANT_URL else 'FALTA'}")
print(f"QDRANT_API_KEY: {'OK' if QDRANT_API_KEY else 'FALTA'}")
print(f"DRIVE_FOLDER_ID: {DRIVE_FOLDER_ID}")

# Verificar variables críticas
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY no está configurada en las variables de entorno")
if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL no está configurada en las variables de entorno")
if not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_API_KEY no está configurada en las variables de entorno")

if API_KEY:
    genai.configure(api_key=API_KEY)

# --- Variables Globales y de Estado ---
# Usaremos una variable global para saber si el sistema ya fue inicializado
IS_INITIALIZED = False
_LAST_ACTIVE: Dict[str, str] = {}  # memoria corta del acto activo

# --- Inicialización del Cliente Qdrant y Modelos ---
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = genai.GenerativeModel(MODEL) if API_KEY else None

# --- Función para generar embeddings (reemplaza ChromaDB embedding function) ---
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Genera embeddings usando Gemini para una lista de textos."""
    if not API_KEY:
        raise ValueError("API_KEY de Gemini no configurada")
    
    embeddings = []
    for i, text in enumerate(texts):
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
            if (i + 1) % 10 == 0:
                print(f"Generados {i + 1}/{len(texts)} embeddings...")
        except Exception as e:
            print(f"Error generando embedding para texto {i}: {e}")
            # Embedding dummy en caso de error
            embeddings.append([0.0] * 768)
    
    return embeddings

def get_query_embedding(query: str) -> List[float]:
    """Genera embedding para una consulta."""
    if not API_KEY:
        raise ValueError("API_KEY de Gemini no configurada")
    
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generando embedding de consulta: {e}")
        return [0.0] * 768

# ── Regex & Claves (Sin cambios) ────────────────────────────────
RES_RE = re.compile(r"\b\d{4,6}-\d{4}\b")
INT_RE = re.compile(r"\b[A-Z]{2}-\d{3,4}\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
LIST_RE = re.compile(r"\b(lista|listado|muestr[ae]|mostrar|dame|ens[eñ]a|ensena)\b", re.I)
HELLO_RE = re.compile(r"\b(hola|buen[oa]s(?:\s*d[ií]as|\s*tardes|\s*noches)?|saludos|qu[eé] tal)\b", re.I)
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

# --- FUNCIÓN DE INICIALIZACIÓN MODIFICADA PARA QDRANT ---
def initialize_rag_system():
    """
    Descarga los PDFs, los procesa y los carga en la base de datos Qdrant.
    Esta función se ejecuta solo una vez al inicio.
    """
    global IS_INITIALIZED
    print("Iniciando sistema RAG con Qdrant...")
    
    try:
        # 1. Verificar si la colección ya existe
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if not collection_exists:
            # Crear la colección con configuración de vectores
            print(f"Creando colección '{COLLECTION_NAME}' en Qdrant...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,  # Tamaño del embedding de Gemini
                    distance=Distance.COSINE
                )
            )
        else:
            # Verificar si ya tiene datos
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            if collection_info.points_count > 0:
                print(f"Colección '{COLLECTION_NAME}' ya existe con {collection_info.points_count} puntos")
                IS_INITIALIZED = True
                return
            else:
                print(f"Colección '{COLLECTION_NAME}' existe pero está vacía, procediendo a cargar datos...")
        
        # 2. Listar archivos PDF en la carpeta de Drive
        pdf_files = list_pdf_files_in_folder(DRIVE_FOLDER_ID)
        if not pdf_files:
            raise RuntimeError(f"No se encontraron archivos PDF en la carpeta de Drive con ID: {DRIVE_FOLDER_ID}")

        all_docs = []
        all_metadatas = []
        
        # 3. Descargar y procesar cada archivo PDF
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
            
            for doc in docs:
                all_docs.append(doc.page_content)
                all_metadatas.append({
                    "source": pdf_name,
                    "page": doc.metadata.get("page", 0),
                    "file_id": pdf_id
                })
            
            # Opcional: borrar el archivo descargado para ahorrar espacio en el contenedor
            os.remove(local_pdf_path)

        if not all_docs:
            raise RuntimeError("No se pudo procesar ningún documento PDF.")

        # 4. Generar embeddings para todos los documentos
        print(f"Generando embeddings para {len(all_docs)} fragmentos...")
        embeddings = get_embeddings_batch(all_docs)
        
        # 5. Preparar puntos para Qdrant (equivalente a ChromaDB col.add)
        points = []
        for i, (doc, metadata, embedding) in enumerate(zip(all_docs, all_metadatas, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "document": doc,  # Equivalente al documento en ChromaDB
                        "metadata": metadata  # Equivalente a metadatas en ChromaDB
                    }
                )
            )
        
        # 6. Insertar los puntos en Qdrant en lotes (equivalente a ChromaDB col.add)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            print(f"Insertado lote {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
        
        IS_INITIALIZED = True
        print("¡Sistema RAG con Qdrant inicializado exitosamente!")
        
    except Exception as e:
        print(f"ERROR FATAL DURANTE LA INICIALIZACIÓN: {e}")
        raise e

# ── Helpers (Tu código original sin cambios) ────────────────────────────────
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

# ── Prompt ficha (Tu código original sin cambios) ───────────────────────────
PROMPT_FICHA = (
    "Usted es **Lexi**, asistente virtual de la División Jurídica de la CGR (Costa Rica).\n"
    "Tono claro y profesional. Responda solo con datos presentes en el 'Contexto'.\n\n"
    "**Contexto relevante:**\n{context}\n\n"
    "**Consulta del usuario:** {query}\n\n"
    "Responda de forma clara y precisa basándose únicamente en el contexto proporcionado."
)

build_prompt = lambda **kw: PROMPT_FICHA.format(
    resol=kw.get("resol") or "desconocido",
    interno=kw.get("interno") or "desconocido",
    sancion=kw.get("sancion") or "sin indicios en metadatos",
    context=_norm(kw.get("context")) or "",
    query=_norm(kw.get("query")) or "",
)

# ── Generación robusta (Tu código original sin cambios) ─────────────────────
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

# ── Modo libre (Tu código original sin cambios) ─────────────────────────────
MSG_INICIAL = (
    "¡Hola! 👋 Con mucho gusto le ayudo. Para buscar un acto final, indíqueme el "
    "**número de resolución** (p. ej. 07685-2025) o el **número interno** (p. ej. DJ-0612)."
)
MSG_DESPEDIDA = "¡Gracias por escribir! Si necesita otra consulta, aquí estaré. 👋"

# ── Router principal (MODIFICADO para Qdrant) ───────────────────
def answer(query: str, k: int = 10, debug: bool = False):
    """
    Función principal que procesa la consulta del usuario.
    Ahora usa Qdrant en lugar de ChromaDB pero mantiene la misma lógica.
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

    # --- Lógica de conversación (tu código original sin cambios) ---
    if GOODBYE_RE.search(t): return MSG_DESPEDIDA, []
    if HELLO_RE.search(t): return MSG_INICIAL, []
    if COURTESY_RE.search(t): return "¡Con mucho gusto! ¿Desea consultar alguna resolución o expediente?", []

    # --- Lógica RAG (MODIFICADA para usar Qdrant en lugar de ChromaDB) ---
    print(f"Realizando consulta a Qdrant con: '{q}'")
    
    try:
        # Generar embedding para la consulta (equivalente a ChromaDB automático)
        query_embedding = get_query_embedding(q)
        
        # Buscar en Qdrant (equivalente a col.query en ChromaDB)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        # Extraer documentos y metadatos (equivalente a ChromaDB response)
        docs = []
        metas = []
        for result in search_results:
            docs.append(result.payload["document"])
            metas.append(result.payload["metadata"])
            if debug:
                print(f"Score: {result.score:.3f}, Source: {result.payload['metadata']['source']}")

        if not docs:
            return "No se encontró información relevante en los documentos para su consulta.", []

        # Construimos un contexto simple con los resultados (mismo que antes)
        context = "\n\n---\n\n".join(docs)
        
        # Generamos la respuesta final (mismo que antes)
        prompt = build_prompt(context=context, query=q)
        final_response = safe_generate(prompt)
        
        if not final_response:
            final_response = "No pude generar una respuesta a partir de la información encontrada."

        return final_response, metas
        
    except Exception as e:
        print(f"Error durante la búsqueda en Qdrant: {e}")
        return "⚠️ Ocurrió un error durante la búsqueda. Por favor, intente de nuevo.", []

# Export para app.py (mismo que antes)
__all__ = ["answer", "GOODBYE_RE"]
