# rag_chain.py (compacto)
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, re, json, time, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from google.api_core.exceptions import ResourceExhausted

# ── Config ─────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
INDEX   = Path("chroma_index")
CATALOG = INDEX / "catalog.json"

if API_KEY:
    genai.configure(api_key=API_KEY)

# Embeddings (usa Gemini si hay API, si no, fallback local)
try:
    emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY, model_name="gemini-embedding-001"
    ) if API_KEY else None
except Exception:
    emb_fn = None

if emb_fn is None:
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

llm = genai.GenerativeModel(MODEL) if API_KEY else None

client = chromadb.PersistentClient(path=str(INDEX))
col    = client.get_collection("resoluciones", embedding_function=emb_fn)

_LAST_ACTIVE: Dict[str, str] = {}  # memoria corta del acto activo

# ── Regex & claves ─────────────────────────────────────────────
RES_RE   = re.compile(r"\b\d{4,6}-\d{4}\b")        # 07685-2025
INT_RE   = re.compile(r"\b[A-Z]{2}-\d{3,4}\b")     # DJ-0612
YEAR_RE  = re.compile(r"\b(19|20)\d{2}\b")
LIST_RE  = re.compile(r"\b(lista|listado|muestr[ae]|mostrar|dame|ens[eñ]a|ensena)\b", re.I)
HELLO_RE = re.compile(r"\b(hola|buen[oa]s(?:\s*d[ií]as|\s*tardes|\s*noches)?|saludos|qu[eé] tal)\b", re.I)
GOODBYE_RE = re.compile(r"\b(ad[ií]os|hasta luego|nos vemos|chao|bye|hasta pronto)\b", re.I)
COURTESY_RE = re.compile(r"\b(gracias|muchas gracias|perfecto|de acuerdo|entendido)\b", re.I)

KEYWORDS_RAG = {
    "resolución","resolucion","acto final","expediente",
    "número interno","numero interno","dj-","pa-","nn",
    "número de resolución","numero de resolución","folio",
}

SANCION_KEYS = {
    "despido sin responsabilidad": re.compile(r"despido\s+sin\s+responsabilidad", re.I),
    "despido con responsabilidad": re.compile(r"despido\s+con\s+responsabilidad", re.I),
    "suspensión":                   re.compile(r"suspensi[oó]n", re.I),
    "inhabilitación":               re.compile(r"inhabilitaci[oó]n", re.I),
    "multa":                        re.compile(r"multa", re.I),
    "archivo":                      re.compile(r"archivo", re.I),
    "apercibimiento":               re.compile(r"apercibimiento", re.I),
}

# ── Helpers ────────────────────────────────────────────────────
_norm = lambda s: " ".join((s or "").split())

def _load_catalog() -> dict:
    if CATALOG.exists():
        with CATALOG.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _sancion_tipo_simple(texto: str | None, tipo: str | None = None) -> str | None:
    if tipo and tipo.strip():
        return tipo
    if not texto:
        return None
    low = texto.lower()
    for etiqueta, patt in SANCION_KEYS.items():
        if patt.search(low):
            return etiqueta
    return None

def _table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    if not rows:
        return "No se encontraron registros que cumplan con la condición solicitada."
    sep = "|".join(["---" for _ in headers])
    out = [" | ".join(headers), sep]
    out += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows]
    return "\n".join(out)

def _list(cat: dict, *, by: str | None = None, year: str | None = None, sancion: str | None = None) -> str:
    rows = []
    patt = SANCION_KEYS.get((sancion or "").lower()) if sancion else None
    for row in cat.values():
        res, interno, sanc = row.get("resolucion"), row.get("interno"), row.get("sancion")
        if year and not ((res or "").endswith(f"-{year}")): continue
        if patt and not patt.search(sanc or ""):           continue
        if by == "res":
            if not res: continue
            rows.append({
                "Número de resolución": res,
                "Número interno": interno or "No consta en el texto",
                "Tipo de sanción": _sancion_tipo_simple(row.get("sancion"), row.get("tipo")) or "No consta en el texto",
                "Fuente": row.get("source"),
            })
        elif by == "int":
            if not interno: continue
            rows.append({
                "Número interno": interno,
                "Número de resolución": res or "No consta en el texto",
                "Tipo de sanción": _sancion_tipo_simple(row.get("sancion"), row.get("tipo")) or "No consta en el texto",
                "Fuente": row.get("source"),
            })
    key = "Número de resolución" if by == "res" else "Número interno"
    rows.sort(key=lambda r: r[key])
    headers = ["Número de resolución", "Número interno", "Tipo de sanción", "Fuente"] if by == "res" else ["Número interno", "Número de resolución", "Tipo de sanción", "Fuente"]
    return _table(rows, headers)

def _pick_idx(metas: List[dict], wanted_res=None, wanted_int=None) -> Optional[int]:
    for i, m in enumerate(metas):
        if wanted_res and m.get("resolucion") == wanted_res: return i
        if wanted_int and m.get("interno") == wanted_int:   return i
    return None

def _group_same_case(docs: List[str], metas: List[dict], top_n: int = 3) -> Tuple[str, dict]:
    if not docs: return "", {}
    base = metas[0]
    key0 = (base.get("resolucion") or "", base.get("interno") or "", base.get("source") or "")
    merged = [docs[0]]
    for j in range(1, len(docs)):
        if len(merged) >= top_n: break
        m = metas[j]
        key = (m.get("resolucion") or "", m.get("interno") or "", m.get("source") or "")
        if key == key0: merged.append(docs[j])
    return "\n\n".join(merged), base

def _disambiguation(metas: List[dict], max_items: int = 6) -> str:
    seen, rows = set(), []
    for m in metas:
        key = (m.get("resolucion") or "—", m.get("interno") or "—", m.get("source") or "—")
        if key in seen: continue
        seen.add(key)
        rows.append({"Número de resolución": key[0], "Número interno": key[1], "Fuente": key[2]})
        if len(rows) >= max_items: break
    return _table(rows, ["Número de resolución", "Número interno", "Fuente"])

def _pick_index_by_meta(metas_list: List[dict], wanted_res=None, wanted_int=None) -> Optional[int]:
    """Devuelve el índice del primer meta que coincide con el # de resolución o # interno."""
    if wanted_res:
        for i, m in enumerate(metas_list):
            if m.get("resolucion") == wanted_res:
                return i
    if wanted_int:
        for i, m in enumerate(metas_list):
            if m.get("interno") == wanted_int:
                return i
    return None

# ── Prompt ficha ───────────────────────────────────────────────
PROMPT_FICHA = (
    "Usted es **Lexi**, asistente virtual de la División Jurídica de la CGR (Costa Rica).\n"
    "Tono claro y profesional. Responda solo con datos presentes en el ‘Contexto’.\n\n"
    "Entregue UNA ficha con este formato EXACTO:\n\n"
    "**Resumen:** Resume la información en 4 renglones\n"
    "**Número de resolución:** N.º ####-AAAA\n"
    "**Número interno:** DJ-####\n"
    "**Procedimiento administrativo:** PA-AAAA####\n"
    "**Fecha:** DD de <mes> de AAAA\n"
    "**Persona investigada:** Nombre completo\n"
    "**Motivo:** <texto>\n"
    "**Resultado (Sanción):**\n"
    "**– Tipo:** [Prisión│Despido con responsabilidad patronal│Despido sin responsabilidad│Suspensión│Inhabilitación│Multa│Apercibimiento│Archivo│Otra]\n"
    "**– Duración o monto:** (si aplica)\n"
    "**– Fundamento legal:** (artículo y norma)\n\n"
    "Reglas:\n- Si falta un dato, escriba exactamente ‘No consta en el texto’.\n"
    "- No incluya fuentes ni anexos.\n"
    "- Si el mensaje del usuario es saludo/despedida, responda breve y no agregue extra.\n\n"
    "Metadatos:\n- Número de resolución: {resol}\n- Número interno: {interno}\n- Indicios de sanción: {sancion}\n\n"
    "Contexto:\n\"\"\"{context}\"\"\"\n\n"
    "Pregunta del usuario:\n\"\"\"{query}\"\"\"\n\n"
    "Redacte la ficha ahora."
)

build_prompt = lambda **kw: PROMPT_FICHA.format(
    resol=kw.get("resol") or "desconocido",
    interno=kw.get("interno") or "desconocido",
    sancion=kw.get("sancion") or "sin indicios en metadatos",
    context=_norm(kw.get("context")) or "",
    query=_norm(kw.get("query")) or "",
)

# ── Generación robusta ─────────────────────────────────────────
def safe_generate(prompt: str, retries: int = 2) -> str:
    if not llm:  # sin API → no LLM
        return ""
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

# ── Modo libre ─────────────────────────────────────────────────
MSG_INICIAL  = (
    "¡Hola! 👋 Con mucho gusto le ayudo. Para buscar un acto final, indíqueme el "
    "**número de resolución** (p. ej. 07685-2025) o el **número interno** (p. ej. DJ-0612)."
)
MSG_DESPEDIDA = "¡Gracias por escribir! Si necesita otra consulta, aquí estaré. 👋"

# ── Router principal ───────────────────────────────────────────
def _is_rag_intent(q: str) -> bool:
    t = q.lower()
    return bool(
        RES_RE.search(q) or INT_RE.search(q) or
        any(k in t for k in KEYWORDS_RAG) or
        (YEAR_RE.search(q) and any(w in t for w in ("dj","resol","acto","pa-")))
    )

def answer(query: str, k: int = 10, debug: bool = False):
    q = (query or "").strip()
    t = q.lower()
    cat = _load_catalog()

    # 0) Frases de cortesía
    if COURTESY_RE.search(t) and not _is_rag_intent(t):
        return "¡Con mucho gusto! ¿Desea consultar alguna resolución o expediente?", []

    # 1) Despedidas explícitas
    if GOODBYE_RE.search(t):
        return MSG_DESPEDIDA, []

    # 2) Saludos que no sean intención RAG
    if HELLO_RE.search(t) and not _is_rag_intent(t):
        return MSG_INICIAL, []

    # 3) Detección de IDs explícitos (uno o varios)
    ids_res = list(dict.fromkeys(RES_RE.findall(t)))
    ids_int = list(dict.fromkeys(INT_RE.findall(t)))
    has_id = bool(ids_res or ids_int)

    # 4) Listas rápidas (solo si no hay ID)
    if not has_id and LIST_RE.search(t) and ("resolucion" in t or "resolución" in t):
        return _list(cat, by="res"), []
    if not has_id and LIST_RE.search(t) and any(x in t for x in ["interno", "despacho", "nn"]):
        return _list(cat, by="int"), []
    if not has_id:
        for etiqueta in SANCION_KEYS:
            if etiqueta in t:
                return _list(cat, by="res", sancion=etiqueta), []
    m_year = YEAR_RE.search(t)
    if not has_id and LIST_RE.search(t) and m_year:
        return _list(cat, by="res", year=m_year.group()), []

    # 5) No es RAG → respuesta libre
    if not _is_rag_intent(t):
        return MSG_INICIAL, []

    # 6) Múltiples IDs → generar una ficha por cada uno
    if len(ids_res) + len(ids_int) > 1:
        fichas, metas_acum = [], []
        for rid in ids_res + ids_int:
            res = col.query(query_texts=[rid], n_results=k, include=["documents", "metadatas", "distances"])
            docs = res.get("documents", [[]])[0] or []
            metas = res.get("metadatas", [[]])[0] or []
            dists = res.get("distances", [[]])[0] or []

            if not docs:
                fichas.append(f"No se encontraron fragmentos para {rid}.")
                continue

            idx = _pick_index_by_meta(metas, wanted_res=rid if rid in ids_res else None, wanted_int=rid if rid in ids_int else None) or 0
            context, meta = _group_same_case(docs[idx:], metas[idx:], top_n=3)
            if not context:
                context, meta = docs[idx], metas[idx]

            resol, interno, sancion = meta.get("resolucion"), meta.get("interno"), meta.get("sancion")
            _LAST_ACTIVE.update({"resol": resol or "", "interno": interno or "", "source": meta.get("source", "")})

            prompt = build_prompt(resol=resol, interno=interno, sancion=sancion, context=context, query=q)
            txt = safe_generate(prompt) or "\n".join([
                "Resumen:",
                f"Número de resolución: {resol or 'No consta en el texto'}",
                f"Número interno: {interno or 'No consta en el texto'}",
                "Procedimiento administrativo: No consta en el texto",
                "Fecha: No consta en el texto",
                "Persona investigada: No consta en el texto",
                "Motivo: No consta en el texto",
                "Resultado (Sanción):",
                f"– Tipo: {_sancion_tipo_simple(sancion) or 'No consta en el texto'}",
                "– Duración o monto (si aplica)",
                "– Fundamento legal: No consta en el texto",
            ])
            if debug:
                txt += "\n\n---\n_Debug_: " + json.dumps({"meta": meta, "rid": rid, "dist": (dists[idx] if dists else None)}, ensure_ascii=False, indent=2)

            fichas.append(txt)
            metas_acum.append(meta)
        return "\n\n---\n\n".join(fichas), metas_acum

    # 7) Caso normal → flujo estándar
    wanted_res = ids_res[0] if ids_res else None
    wanted_int = ids_int[0] if ids_int else None

    res = col.query(query_texts=[q], n_results=k, include=["documents", "metadatas", "distances"])
    docs, metas, dists = res.get("documents", [[]])[0] or [], res.get("metadatas", [[]])[0] or [], res.get("distances", [[]])[0] or [math.inf] * k

    if not docs:
        if _LAST_ACTIVE:
            wanted_res = _LAST_ACTIVE.get("resol")
            wanted_int = _LAST_ACTIVE.get("interno")
        else:
            return "No encontrado en la base de resoluciones.", []

    idx = _pick_index_by_meta(metas, wanted_res=wanted_res, wanted_int=wanted_int) or 0

    # 8) Desambiguación
    if not has_id and idx == 0:
        uniq = []
        for m in metas[:min(8, len(metas))]:
            key = (m.get("resolucion") or "", m.get("interno") or "", m.get("source") or "")
            if key not in uniq:
                uniq.append(key)
        if len(uniq) > 1:
            return ("Encontré varios actos finales posibles. Indique el **Número de resolución** o **Número interno** exacto de esta lista:\n\n" + _disambiguation(metas)), []

    # 9) Construcción de contexto
    context, meta = _group_same_case(docs[idx:], metas[idx:], top_n=3)
    if not context:
        context, meta = docs[idx], metas[idx]

    resol, interno, sancion = meta.get("resolucion"), meta.get("interno"), meta.get("sancion")
    _LAST_ACTIVE.update({"resol": resol or "", "interno": interno or "", "source": meta.get("source", "")})

    # 10) Generación de ficha
    prompt = build_prompt(resol=resol, interno=interno, sancion=sancion, context=context, query=q)
    txt = safe_generate(prompt) or "\n".join([
        "Resumen:",
        f"Número de resolución: {resol or 'No consta en el texto'}",
        f"Número interno: {interno or 'No consta en el texto'}",
        "Procedimiento administrativo: No consta en el texto",
        "Fecha: No consta en el texto",
        "Persona investigada: No consta en el texto",
        "Motivo: No consta en el texto",
        "Resultado (Sanción):",
        f"– Tipo: {_sancion_tipo_simple(sancion) or 'No consta en el texto'}",
        "– Duración o monto (si aplica)",
        "– Fundamento legal: No consta en el texto",
    ])

    if debug:
        txt += "\n\n---\n_Debug_: " + json.dumps({"activo": _LAST_ACTIVE.copy(), "meta": meta, "dist": dists[idx] if dists else None}, ensure_ascii=False, indent=2)

    return txt, [meta]

# Export para app.py
__all__ = ["answer", "GOODBYE_RE"]