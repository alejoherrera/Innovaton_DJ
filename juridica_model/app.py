# app.py
import base64, mimetypes
import gradio as gr
from pathlib import Path
from rag_chain import answer

# ─── Colores y rutas ────────────────────────────────────────────
ORG_COLOR = "#284293"      # azul CGR
BG_CHAT   = "#f2f4ff"      # gris clarito
TXT_CHAT  = "#284293"
LOGO      = Path("static/Logotipo-CGR-transp.png")  # ruta relativa

# ─── Utilidad: logo → data-URI para <img> ───────────────────────
def to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0]
    b64  = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

# ─── CSS ────────────────────────────────────────────────────────
custom_css = f"""
:root {{
  --bg-chat:  {BG_CHAT};
  --txt-chat: {TXT_CHAT};
  --prim:     {ORG_COLOR};
}}

body, .gradio-container {{ background:white; font-family:'Poppins',sans-serif; }}

#panel {{ max-width:600px; margin:auto; }}

h1.title {{
  color:var(--prim); font-weight:600; text-align:center;
  margin:1rem 0 2rem;
}}

#chatbot, #chatbot *, #chatbot .message, #chatbot .bubble {{
  background:var(--bg-chat) !important;
  color:var(--txt-chat) !important;
  border:none !important;
}}
#chatbot .label {{ display:none !important; }}   /* quita “Chatbot” */

#input_zone textarea {{
  background:var(--prim) !important;
  color:white !important;
  text-align:center; font-weight:600;
  border-radius:8px; border:none;
}}
#input_zone textarea::placeholder {{ color:white; opacity:1; }}
#input_zone .gr-button{{ display:none; }}        /* ocultamos el botón */

#note {{ font-size:12px; text-align:center; color:var(--prim); margin-top:12px; }}
"""

# ─── Lógica de conversación ────────────────────────────────────
def chat_fn(msg, hist):
    hist.append((msg, "⌛ Consultando…"))
    yield "", hist
    resp, _ = answer(msg, k=10)
    hist[-1] = (msg, resp)
    yield "", hist

# ─── Interfaz ──────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="RAG | Resoluciones DJ") as demo:
    with gr.Column(elem_id="panel"):
        gr.HTML(f"""
        <div style='display:flex; justify-content:center; margin-top:32px;'>
            <img src="{to_data_uri(LOGO)}" style="max-height:96px;">
        </div>""")
        gr.Markdown("<h1 class='title'>RAG&nbsp; |&nbsp; Resoluciones de acto final DJ</h1>")

        chatbot = gr.Chatbot(type="tuples", elem_id="chatbot")

        with gr.Row(elem_id="input_zone"):
            txt = gr.Textbox(
                placeholder="¡Escribe tu consulta aquí!",
                show_label=False, lines=1, container=False
            )            

        gr.HTML("<div id='note'>Nota: Consulte sobre el número de resolución o número interno del acto final que desea buscar</div>")

        txt.submit(chat_fn, [txt, chatbot], [txt, chatbot])        

if __name__ == '__main__':
    demo.launch()