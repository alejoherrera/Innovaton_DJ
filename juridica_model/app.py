import base64
import mimetypes
from pathlib import Path
import gradio as gr
import os  # ¡Importante! Necesitamos 'os' para el puerto.
from rag_chain import answer, GOODBYE_RE

ORG = "#284293"  # azul CGR
# LOGO = Path("static/Logotipo-CGR-transp.png") # Comentado temporalmente para evitar FileNotFoundError
SUGERENCIA_HTML = (
    "Sugerencia: también puede pedir "
    "lista de resoluciones 2024 o "
    "lista con despido sin responsabilidad."
)

# def to_data_uri(p: Path) -> str:
#     mime = mimetypes.guess_type(p)[0]
#     return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

CSS = f"""
:root {{ --prim:{ORG}; }}
body, .gradio-container {{ background:white; font-family:'Poppins',sans-serif; }}
#wrap {{ max-width:680px; margin: 24px auto 40px; }}
#logo img {{ height:96px; display:block; margin:0 auto; }}
#title {{ color:var(--prim); text-align:center; font-weight:600; margin:16px 0 8px; }}
#chatbot, #chatbot * {{ background:#f2f4ff !important; color:#284293 !important; border:none !important; }}
#chatbot .label {{ display:none !important; }}
#inbox textarea {{ background:var(--prim)!important; color:white!important; font-weight:600; text-align:center; border:none; border-radius:10px; }}
#inbox .gr-button {{ display:none; }} /* ocultar botón: solo Enter */
#note {{ font-size:12px; text-align:center; color:#ffa76c; margin-top:10px; font-weight: bold;}}
"""

def chat_fn(msg, hist):
    if not hist:
        hist = [("", "¡Hola! 👋 Soy **Lexi** de la División Jurídica.\nIndíqueme el **número de resolución** (p. ej. 07685-2025) o el **número interno** (p. ej. DJ-0612). También puedo conversar en general.")]
        yield "", hist
    hist.append((msg, "⌛ Consultando…")); yield "", hist
    try:
        resp, _ = answer(msg, k=10)
    except Exception as e:
        print(f"Error en chat_fn: {e}") # Añadimos un print para ver el error en los logs
        resp = "⚠️ Ocurrió un error procesando su consulta. Por favor, intente de nuevo."
    hist[-1] = (msg, resp); yield "", hist

with gr.Blocks(css=CSS, title="RAG | Resoluciones DJ") as demo:
    with gr.Column(elem_id="wrap"):
        # gr.HTML(f"<div id='logo'><img src='{to_data_uri(LOGO)}' /></div>") # Comentado temporalmente
        gr.Markdown("<h1 id='title'>RAG&nbsp; |&nbsp; Resoluciones de acto final (DJ)</h1>")
        chat = gr.Chatbot(type="tuples", elem_id="chatbot")
        with gr.Row(elem_id="inbox"):
            txt = gr.Textbox(placeholder="Escriba su consulta… (p. ej., 07685-2025 o DJ-0612)", show_label=False, lines=1, container=False)
        gr.HTML(f"<div id='note'>{SUGERENCIA_HTML}</div>")
        txt.submit(chat_fn, [txt, chat], [txt, chat])

if __name__ == "__main__":
    # Esta es la línea modificada para que funcione en Cloud Run
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get('PORT', 8080)))

