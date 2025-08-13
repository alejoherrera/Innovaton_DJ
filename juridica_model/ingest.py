# ingest.py
from dotenv import load_dotenv
import os, time, re, json
from pathlib import Path
from PyPDF2 import PdfReader
from PyPDF2.errors import EmptyFileError, PdfReadError
import chromadb
from chromadb.utils import embedding_functions
from google.api_core.exceptions import ResourceExhausted
from drive_utils import list_pdfs, download_file

# =========================
# 1. CONFIGURACIÓN GENERAL
# =========================
load_dotenv()  # lee .env

DATA_DIR    = Path("pdfs")
INDEX_DIR   = Path("chroma_index")
TMP_DIR     = Path("pdfs_tmp")
FAILED_DIR  = Path("pdfs_fallidos")
CATALOG_PATH = INDEX_DIR / "catalog.json"
catalog: dict[str, dict] = {}

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
FAILED_DIR.mkdir(exist_ok=True)

FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
if not FOLDER_ID:
    raise ValueError("🚨 Falta DRIVE_FOLDER_ID en .env")

# =========================
# 2. UTILIDADES PDF
# =========================
def is_valid_pdf(path: Path) -> bool:
    """Devuelve True si el archivo existe, pesa >0 y PyPDF2 puede abrirlo."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        r = PdfReader(str(path))
        return len(r.pages) > 0
    except (EmptyFileError, PdfReadError, OSError):
        return False

def pdf_to_text(path: Path) -> str:
    """Extrae texto de todas las páginas; si falla, lanza excepción."""
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

# =========================
# 3. REGEX PARA METADATOS
# =========================
# Números de resolución e interno (tus reglas)
res_regex = re.compile(r"\b\d{4,6}-\d{4}\b")      # 07685-2025
int_regex = re.compile(r"\b[A-Z]{2}-\d{3,4}\b")   # DJ-0612
pa_regex = re.compile(r"\b(?:CGR-)?PA-\d{8,10}\b", re.I)

# Sanitiza sanciones + detecta "archivo"
sancion_regex = re.compile(
    r"(separaci[oó]n del cargo[^\n]*?|despido[^\n]*?responsabilidad[^\n]*?|"
    r"suspensi[oó]n[^\n]*?(d[ií]as|meses|años)|inhabilitaci[oó]n[^\n]*?años?|"
    r"prohibici[oó]n de ingreso[^\n]*?|multa[^\n]*?¢[\d\.]+|archivo)",
    re.IGNORECASE | re.DOTALL
)

TIPO_PATTERNS = {
    "despido sin responsabilidad": r"despido\s+sin\s+responsabilidad",
    "despido con responsabilidad": r"despido\s+con\s+responsabilidad",
    "suspensión":                   r"suspensi[oó]n",
    "inhabilitación":               r"inhabilitaci[oó]n",
    "multa":                        r"\bmulta\b",
    "archivo":                      r"\barchivo\b",
    "apercibimiento":               r"apercibimiento",
}

def sancion_a_tipo(s: str | None) -> str | None:
    if not s:
        return None
    low = s.lower()
    for tipo, patt in TIPO_PATTERNS.items():
        if re.search(patt, low, re.I):
            return tipo
    return None

def scan_pdf_metadata(pdf_path: Path) -> dict:
    text = pdf_to_text(pdf_path)
    meta = {}
    if m := res_regex.search(text):
        meta["resolucion"] = m.group()
    if m := int_regex.search(text):
        meta["interno"] = m.group()
    if m := pa_regex.search(text):
        meta["pa"] = m.group()
    if m := sancion_regex.search(text):
        sanc = " ".join(m.group().split())
        meta["sancion"] = sanc
    return meta

# =========================
# 4. DESCARGA DESDE DRIVE
# =========================
MAX_DL_RETRIES = 2  # total 3 intentos

for f in list_pdfs(FOLDER_ID):
    final_dst = DATA_DIR / f["name"]
    tmp_dst   = TMP_DIR / f["name"]
    bad_dst   = FAILED_DIR / f["name"]

    # ya marcado como fallido → no reintentar en esta corrida
    if bad_dst.exists():
        print(f"↩️  Ya marcado como inválido: {f['name']}. Lo salto.")
        continue

    # ya descargado y válido → nada que hacer
    if final_dst.exists() and is_valid_pdf(final_dst):
        continue

    # descargar SIEMPRE a TMP; mover a destino solo si es válido
    ok = False
    for attempt in range(1, MAX_DL_RETRIES + 2):
        print(f"⬇️  Bajando {f['name']} (intento {attempt})")
        try:
            if tmp_dst.exists():
                try: tmp_dst.unlink()
                except OSError: pass

            download_file(f["id"], tmp_dst)
            if is_valid_pdf(tmp_dst):
                ok = True
                break
        except Exception as e:
            print(f"⚠️  Error descargando {f['name']}: {e}")

    if ok:
        if final_dst.exists():
            try: final_dst.unlink()
            except OSError: pass
        tmp_dst.replace(final_dst)  # mover atómico
    else:
        print(f"🚫  Archivo inválido tras reintentos: {f['name']}. Lo marco como fallido.")
        if bad_dst.exists():
            try: bad_dst.unlink()
            except OSError: pass
        if tmp_dst.exists():
            tmp_dst.replace(bad_dst)

# =========================
# 5. EMBEDDINGS DE GEMINI
# =========================
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-embedding-001"
)

client = chromadb.PersistentClient(path=str(INDEX_DIR))
col = client.get_or_create_collection("resoluciones", embedding_function=emb_fn)

def chunkify_text(text: str, chunk=1800, overlap=200):
    tokens = text.split()
    i = 0
    while i < len(tokens):
        yield " ".join(tokens[i:i + chunk])
        i += max(1, chunk - overlap)

# =========================
# 6. INDEXACIÓN + CATALOGO
# =========================
def safe_add(chunk: str, meta: dict, uid: str, max_retries=3, sleep_secs=65) -> bool:
    retries = 0
    while retries <= max_retries:
        try:
            col.add(documents=[chunk], metadatas=[meta], ids=[uid])
            return True
        except ResourceExhausted:
            if retries == max_retries:
                print(f"🚫 No se pudo indexar {uid} por límite de cuota. Lo salto.")
                return False
            retries += 1
            print(f"⏳ Cuota agotada; espero {sleep_secs}s… (reintento {retries}/{max_retries})")
            time.sleep(sleep_secs)
        except Exception as e:
            print(f"⚠️ Error indexando {uid}: {e}. Lo salto.")
            return False

pendientes = []
for pdf in sorted(DATA_DIR.glob("*.pdf")):
    if not is_valid_pdf(pdf):
        print(f"⚠️  PDF inválido detectado durante indexación: {pdf.name}. Lo salto.")
        continue

    try:
        # Extrae metadatos (y de paso lee el texto para chunkear)
        text = pdf_to_text(pdf)
        doc_meta = scan_pdf_metadata(pdf)
    except (EmptyFileError, PdfReadError, OSError) as e:
        print(f"⚠️  No se pudo procesar {pdf.name}: {e}. Lo salto.")
        continue
    except Exception as e:
        print(f"⚠️  Error inesperado leyendo {pdf.name}: {e}. Lo salto.")
        continue

    # ► Catálogo (1 fila por PDF)
    catalog[pdf.name] = {
        "source": pdf.name,
        "resolucion": doc_meta.get("resolucion"),
        "interno": doc_meta.get("interno"),
        "pa": doc_meta.get("pa"),                          
        "tipo": sancion_a_tipo(doc_meta.get("sancion")),
    }

    # Genera pendientes por chunk
    for n, chunk in enumerate(chunkify_text(text, chunk=1800, overlap=200)):
        uid = f"{pdf.name}_{n}"
        if col.get(ids=[uid], include=[])["ids"]:
            continue

        # construir metadatos sin 'sancion'
        meta_chunk = {
            "source": pdf.name,
            "chunk": n,
            "resolucion": doc_meta.get("resolucion"),
            "interno": doc_meta.get("interno"),
            "pa": doc_meta.get("pa"),
            "tipo": sancion_a_tipo(doc_meta.get("sancion")),
        }
        pendientes.append((
            uid, chunk,
            {**doc_meta, "source": pdf.name, "chunk": n,
            "tipo": sancion_a_tipo(doc_meta.get("sancion"))}
        ))


if not pendientes:
    print("✅ Base vectorial actualizada (no había nada nuevo)")
else:
    print(f"🧩 Chunks pendientes por indexar: {len(pendientes)}")
    for uid, chunk, meta in pendientes:
        safe_add(chunk, meta, uid)
    print("✅ Base vectorial actualizada")

# Guarda el catálogo al final
INDEX_DIR.mkdir(exist_ok=True)
with CATALOG_PATH.open("w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

print(f"✅ Catálogo actualizado en {CATALOG_PATH}")