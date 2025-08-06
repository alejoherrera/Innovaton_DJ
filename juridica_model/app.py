# app.py
import gradio as gr
from rag_chain import answer

# ─── Colores y rutas ────────────────────────────────────────────
ORG_COLOR = "#36499b"   # azul CGR (título, spinner)
BG_CHAT   = "#f2f4ff"   # gris muy claro
TXT_CHAT  = "#284293"   # texto principal
LOGO      = "static/Logotipo-CGR-blanco-transp.png"

# ─── Estilos personalizados ────────────────────────────────────
custom_css = f"""
:root {{
  --bg:  {BG_CHAT};
  --txt: {TXT_CHAT};
}}
.gradio-container {{background:white; font-family:'Poppins', sans-serif}}
#panel {{max-width:600px; margin:auto; border:1px solid #d9dce2; padding:1rem}}
h1.title {{color:{ORG_COLOR}; text-align:center; margin:1rem 0 2rem}}

/* Fuerza modo claro incluso si el usuario navega en dark-mode */
body[data-mode="dark"] .gradio-container {{ background:white !important; }}

/* Chatbot (contenedor y burbujas) */
#chatbot{{
  background:var(--bg) !important;      /* gris claro */
  color:var(--txt) !important;
}}
#chatbot .message{{
  background:transparent !important;    /* quita burbujas negras */
  color:inherit !important;
}}

/* Caja de entrada */
textarea, input[type="text"]{{
  background:var(--bg) !important;      /* mismo gris claro */
  color:var(--txt) !important;
  border:1px solid #ccd0e0;
}}

"""

# ─── Función de conversación ────────────────────────────────────
def chat_fn(message, history):
    # muestra mensaje de “pensando…”
    history.append((message, "⌛ Canalizando su solicitud, por favor espere…"))
    yield "", history

    # genera la respuesta (sin fuentes)
    resp, _ = answer(message, k=10)
    history[-1] = (message, resp)
    yield "", history

# ─── Construcción de la interfaz ────────────────────────────────
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="panel"):
        with gr.Row():
            gr.Image(LOGO, show_label=False, height=80)
            gr.Markdown(
                f"<div style='font-size:1.4rem; font-weight:700; "
                f"color:{ORG_COLOR}; line-height:1.1;'>Gestión Documental<br>"
                "Institucional</div>"
            )

        gr.Markdown("<h1 class='title'>RAG | Resoluciones de acto final DJ</h1>")

        chatbot = gr.Chatbot(type="tuples", elem_id="chatbot")

        msg = gr.Textbox(
            placeholder="¡Escriba su consulta aquí!",
            show_label=False,
            container=False
        )
        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()