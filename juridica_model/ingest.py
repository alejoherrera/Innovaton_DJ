# ingest.py
from dotenv import load_dotenv
import os, time, re
from pathlib import Path
from tqdm import tqdm
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from google.api_core.exceptions import ResourceExhausted
from drive_utils import list_pdfs, download_file

# =========================
# 1. CONFIGURACIÓN GENERAL
# =========================
load_dotenv()                                  # lee .env
DATA_DIR  = Path("pdfs")
INDEX_DIR = Path("chroma_index")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
if not FOLDER_ID:
    raise ValueError("🚨 Falta DRIVE_FOLDER_ID en .env")

# =========================
# 2. REGEX PARA METADATOS
# =========================
res_regex = re.compile(r"\b\d{4,6}-\d{4}\b")      # 07685-2025
int_regex = re.compile(r"\b[A-Z]{2}-\d{3,4}\b")   # DJ-0612

sancion_regex = re.compile(
    r"(separaci[oó]n del cargo[^\n]*?|despido[^\n]*?responsabilidad[^\n]*?|"
    r"suspensi[oó]n[^\n]*?(d[ií]as|meses|años)|inhabilitaci[oó]n[^\n]*?años?|"
    r"prohibici[oó]n de ingreso[^\n]*?|multa[^\n]*?¢[\d\.]+)",
    re.IGNORECASE | re.DOTALL
)

def scan_pdf_metadata(pdf_path: Path) -> dict:
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)
    meta = {}
    if m := res_regex.search(text):
        meta["resolucion"] = m.group()
    if m := int_regex.search(text):
        meta["interno"] = m.group()
    # NUEVO ► sanción
    if m := sancion_regex.search(text):
        # Une líneas y espacios sobrantes
        sanc = " ".join(m.group().split())
        meta["sancion"] = sanc
    return meta

# =========================
# 3. DESCARGAR NUEVOS PDF
# =========================
for f in list_pdfs(FOLDER_ID):
    dst = DATA_DIR / f["name"]
    if not dst.exists():
        print("⬇️  Bajando", f["name"])
        download_file(f["id"], dst)

# =========================
# 4. FUNCIÓN DE TROCEO
# =========================
def pdf_to_chunks(pdf_path: Path, chunk=1800, overlap=200):
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)
    tokens = text.split()
    i = 0
    while i < len(tokens):
        yield " ".join(tokens[i:i + chunk])
        i += chunk - overlap

# =========================
# 5. EMBEDDINGS DE GEMINI
# =========================
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-embedding-001"
)

client = chromadb.PersistentClient(path=str(INDEX_DIR))
col = client.get_or_create_collection("resoluciones", embedding_function=emb_fn)

# helper para reintentar si se agota la cuota
def safe_add(chunk: str, meta: dict, uid: str):
    while True:
        try:
            col.add(documents=[chunk], metadatas=[meta], ids=[uid])
            break
        except ResourceExhausted:
            print("⏳ Cuota agotada; espero 65 s…")
            time.sleep(65)

# =========================
# 6. INDEXACIÓN
# =========================
for pdf in DATA_DIR.glob("*.pdf"):
    doc_meta = scan_pdf_metadata(pdf)           # se calcula 1 vez por PDF
    for n, chunk in enumerate(pdf_to_chunks(pdf, chunk=1800, overlap=200)):
        uid = f"{pdf.name}_{n}"
        if not col.get(ids=[uid], include=[])["ids"]:
            meta = {**doc_meta, "source": pdf.name, "chunk": n}
            safe_add(chunk, meta, uid)

print("✅ Base vectorial actualizada")
