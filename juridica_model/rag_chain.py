# rag_chain.py (Versión final con manejo robusto de errores)
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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Importamos las funciones para descargar desde Drive
from drive_utils import download_file_from_drive, list_pdf_files_in_folder

# Al inicio de rag_chain.py, después de todos los imports
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Variables .env cargadas")
except ImportError:
    print("ℹ️ python-dotenv no instalado, usando variables del sistema")


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
IS_INITIALIZED = False
_LAST_ACTIVE: Dict[str, str] = {}

# --- Inicialización del Cliente Qdrant y Modelos ---
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = genai.GenerativeModel(MODEL) if API_KEY else None

# --- Función para generar embeddings ---
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

# ── Regex & Claves ────────────────────────────────
RES_RE = re.compile(r"\b\d{4,6}-\d{4}\b")
INT_RE = re.compile(r"\b[A-Z]{2}-\d{3,4}\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
LIST_RE = re.compile(r"\b(lista|listado|muestr[ae]|mostrar|dame|ens[eñ]a|ensena)\b", re.I)
HELLO_RE = re.compile(r"\b(hola|buen[oa]s(?:\s*d[ií]as|\s*tardes|\s*noches)?|saludos|qu[eé] tal)\b", re.I)
GOODBYE_RE = re.compile(r"\b(ad[ií]os|hasta luego|nos vemos|chao|bye|hasta pronto)\b", re.I)
COURTESY_RE = re.compile(r"\b(gracias|muchas gracias|perfecto|de acuerdo|entendido)\b", re.I)

SANCION_KEYS = {
    "despido sin responsabilidad": re.compile(r"despido\s+sin\s+responsabilidad", re.I),
    "despido con responsabilidad": re.compile(r"despido\s+con\s+responsabilidad", re.I),
    "suspensión": re.compile(r"suspensi[oó]n", re.I),
    "inhabilitación": re.compile(r"inhabilitaci[oó]n", re.I),
    "multa": re.compile(r"multa", re.I),
    "archivo": re.compile(r"archivo", re.I),
    "apercibimiento": re.compile(r"apercibimiento", re.I),
}

def process_new_files(all_pdf_files: List[dict], new_file_names: set):
    """Procesa solo los archivos nuevos y los agrega a Qdrant."""
    print("🔄 Procesando archivos nuevos...")
    
    new_docs = []
    new_metadatas = []
    processed_count = 0
    
    for pdf_info in all_pdf_files:
        pdf_name = pdf_info['name']
        if pdf_name not in new_file_names:
            continue  # Saltar archivos ya procesados
            
        pdf_id = pdf_info['id']
        print(f"📄 Procesando nuevo archivo: {pdf_name}")
        
        local_pdf_path = download_file_from_drive(pdf_id, pdf_name)
        if not local_pdf_path or not os.path.exists(local_pdf_path):
            print(f"⚠️ No se pudo descargar: {pdf_name}")
            continue
            
        if os.path.getsize(local_pdf_path) == 0:
            print(f"⚠️ Archivo vacío: {pdf_name}")
            os.remove(local_pdf_path)
            continue
        
        try:
            loader = PyPDFLoader(local_pdf_path)
            pages = loader.load_and_split()
            
            if not pages:
                print(f"⚠️ No se pudieron extraer páginas: {pdf_name}")
                continue
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            docs = text_splitter.split_documents(pages)
            
            valid_chunks = 0
            for doc in docs:
                if doc.page_content.strip():
                    new_docs.append(doc.page_content)
                    new_metadatas.append({
                        "source": pdf_name,
                        "page": doc.metadata.get("page", 0),
                        "file_id": pdf_id
                    })
                    valid_chunks += 1
            
            if valid_chunks > 0:
                print(f"✅ {pdf_name}: {valid_chunks} chunks nuevos")
                processed_count += 1
            
        except Exception as e:
            print(f"❌ Error procesando {pdf_name}: {e}")
        finally:
            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
    
    # Insertar nuevos documentos en Qdrant
    if new_docs:
        print(f"📊 Insertando {len(new_docs)} chunks nuevos en Qdrant...")
        
        embeddings = get_embeddings_batch(new_docs)
        points = []
        
        for doc, metadata, embedding in zip(new_docs, new_metadatas, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"document": doc, "metadata": metadata}
                )
            )
        
        # Insertar en lotes
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
        
        print(f"✅ {processed_count} archivos nuevos procesados exitosamente!")
    else:
        print("⚠️ No se encontraron chunks válidos en los archivos nuevos")

# --- FUNCIÓN DE INICIALIZACIÓN CON DETECCIÓN DE ARCHIVOS NUEVOS ---
def initialize_rag_system():
    """
    Descarga los PDFs, los procesa y los carga en Qdrant.
    Maneja archivos corruptos/vacíos de forma robusta.
    """
    global IS_INITIALIZED
    print("Iniciando sistema RAG con Qdrant...")
    
    try:
        # 1. Verificar si la colección ya existe
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if not collection_exists:
            print(f"Creando colección '{COLLECTION_NAME}' en Qdrant...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            # Verificar si hay archivos nuevos
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            if collection_info.points_count > 0:
                print(f"Colección '{COLLECTION_NAME}' ya existe con {collection_info.points_count} puntos")
                
                # Verificar si hay archivos nuevos para procesar
                print("Verificando archivos nuevos...")
                pdf_files = list_pdf_files_in_folder(DRIVE_FOLDER_ID)
                
                # Obtener archivos ya procesados
                existing_files = set()
                try:
                    # Buscar todos los puntos existentes para obtener los nombres de archivos
                    scroll_result = qdrant_client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=10000,  # Ajustar según necesidades
                        with_payload=True
                    )
                    
                    for point in scroll_result[0]:
                        source = point.payload.get("metadata", {}).get("source", "")
                        if source:
                            existing_files.add(source)
                    
                    print(f"Archivos ya procesados: {len(existing_files)}")
                    
                except Exception as e:
                    print(f"Error obteniendo archivos existentes: {e}")
                    existing_files = set()
                
                # Encontrar archivos nuevos
                current_files = {pdf['name'] for pdf in pdf_files}
                new_files = current_files - existing_files
                
                if new_files:
                    print(f"🆕 Archivos nuevos detectados: {len(new_files)}")
                    for new_file in sorted(new_files):
                        print(f"  - {new_file}")
                    
                    # Procesar solo archivos nuevos
                    process_new_files(pdf_files, new_files)
                else:
                    print("✅ No hay archivos nuevos para procesar")
                
                IS_INITIALIZED = True
                return
        
        # 2. Listar archivos PDF
        pdf_files = list_pdf_files_in_folder(DRIVE_FOLDER_ID)
        if not pdf_files:
            raise RuntimeError(f"No se encontraron archivos PDF en la carpeta: {DRIVE_FOLDER_ID}")

        all_docs = []
        all_metadatas = []
        processed_files = 0
        
        # 3. Procesar cada archivo con manejo robusto de errores
        for pdf_info in pdf_files:
            pdf_id = pdf_info['id']
            pdf_name = pdf_info['name']
            print(f"Procesando: {pdf_name}")
            
            local_pdf_path = download_file_from_drive(pdf_id, pdf_name)
            if not local_pdf_path:
                print(f"⚠️ No se pudo descargar: {pdf_name}")
                continue

            # Verificar que el archivo existe y no está vacío
            if not os.path.exists(local_pdf_path):
                print(f"⚠️ Archivo no existe: {pdf_name}")
                continue
                
            if os.path.getsize(local_pdf_path) == 0:
                print(f"⚠️ Archivo vacío (0 bytes): {pdf_name}")
                os.remove(local_pdf_path)
                continue
            
            try:
                # Intentar cargar y procesar el PDF
                loader = PyPDFLoader(local_pdf_path)
                pages = loader.load_and_split()
                
                if not pages:
                    print(f"⚠️ No se pudieron extraer páginas: {pdf_name}")
                    continue
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                docs = text_splitter.split_documents(pages)
                
                if not docs:
                    print(f"⚠️ No se pudieron crear chunks: {pdf_name}")
                    continue
                
                # Solo agregar chunks con contenido real
                valid_chunks = 0
                for doc in docs:
                    if doc.page_content.strip():
                        all_docs.append(doc.page_content)
                        all_metadatas.append({
                            "source": pdf_name,
                            "page": doc.metadata.get("page", 0),
                            "file_id": pdf_id
                        })
                        valid_chunks += 1
                
                if valid_chunks > 0:
                    print(f"✅ {pdf_name}: {valid_chunks} chunks procesados")
                    processed_files += 1
                else:
                    print(f"⚠️ {pdf_name}: sin contenido válido")
                
            except Exception as e:
                print(f"❌ Error procesando {pdf_name}: {e}")
            finally:
                # Siempre limpiar el archivo temporal
                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)

        # 4. Verificar que tenemos documentos para procesar
        if not all_docs:
            print("⚠️ No se procesaron documentos válidos, creando colección vacía")
            IS_INITIALIZED = True
            return
        
        print(f"📊 Resumen: {processed_files} archivos procesados, {len(all_docs)} chunks totales")

        # 5. Generar embeddings
        print("Generando embeddings...")
        embeddings = get_embeddings_batch(all_docs)
        
        # 6. Insertar en Qdrant
        points = []
        for i, (doc, metadata, embedding) in enumerate(zip(all_docs, all_metadatas, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"document": doc, "metadata": metadata}
                )
            )
        
        # Insertar en lotes
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"Lote {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size} insertado")
        
        IS_INITIALIZED = True
        print("✅ Sistema RAG inicializado exitosamente!")
        
    except Exception as e:
        print(f"❌ ERROR FATAL DURANTE LA INICIALIZACIÓN: {e}")
        raise e

# ── Helpers ────────────────────────────────
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

# ── Prompt ───────────────────────────
PROMPT_FICHA = (
    "Usted es **Lexi**, asistente virtual de la División Jurídica de la CGR (Costa Rica).\n"
    "Tono claro y profesional. Responda solo con datos presentes en el 'Contexto'.\n\n"
    "**Contexto relevante:**\n{context}\n\n"
    "**Consulta del usuario:** {query}\n\n"
    "Responda de forma clara y precisa basándose únicamente en el contexto proporcionado."
)

build_prompt = lambda **kw: PROMPT_FICHA.format(
    context=_norm(kw.get("context")) or "",
    query=_norm(kw.get("query")) or "",
)

# ── Generación robusta ─────────────────────
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

# ── Mensajes ─────────────────────────────
MSG_INICIAL = (
    "¡Hola! 👋 Con mucho gusto le ayudo. Para buscar un acto final, indíqueme el "
    "**número de resolución** (p. ej. 07685-2025) o el **número interno** (p. ej. DJ-0612)."
)
MSG_DESPEDIDA = "¡Gracias por escribir! Si necesita otra consulta, aquí estaré. 👋"

# ── Router principal ───────────────────
def answer(query: str, k: int = 10, debug: bool = False):
    """Función principal que procesa la consulta del usuario."""
    global IS_INITIALIZED
    
    if not IS_INITIALIZED:
        try:
            initialize_rag_system()
        except Exception as e:
            print(f"ERROR FATAL DURANTE LA INICIALIZACIÓN: {e}")
            return "⚠️ Lo siento, el sistema no pudo iniciarse correctamente. Por favor, contacte al administrador.", []

    q = (query or "").strip()
    t = q.lower()

    # Lógica de conversación
    if GOODBYE_RE.search(t): return MSG_DESPEDIDA, []
    if HELLO_RE.search(t): return MSG_INICIAL, []
    if COURTESY_RE.search(t): return "¡Con mucho gusto! ¿Desea consultar alguna resolución o expediente?", []

    # Búsqueda en Qdrant
    print(f"Consultando Qdrant: '{q}'")
    
    try:
        query_embedding = get_query_embedding(q)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        docs = []
        metas = []
        for result in search_results:
            docs.append(result.payload["document"])
            metas.append(result.payload["metadata"])

        if not docs:
            return "No se encontró información relevante en los documentos para su consulta.", []

        context = "\n\n---\n\n".join(docs)
        prompt = build_prompt(context=context, query=q)
        final_response = safe_generate(prompt)
        
        if not final_response:
            final_response = "No pude generar una respuesta a partir de la información encontrada."

        return final_response, metas
        
    except Exception as e:
        print(f"Error durante la búsqueda: {e}")
        return "⚠️ Ocurrió un error durante la búsqueda. Por favor, intente de nuevo.", []

# Export para app.py
__all__ = ["answer", "GOODBYE_RE"]
