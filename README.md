# 🧠 RAG | Resoluciones de Acto Final – División Jurídica CGR

Este proyecto convierte el repositorio de actos finales en un sistema RAG (Retrieval-Augmented Generation) que:

- Sincroniza los PDF desde una carpeta oficial de Google Drive.

- Indice su contenido en ChromaDB usando embeddings de Gemini.

- Expone un chat Gradio (“Lexi”) con respuestas amistosas y precisas para el equipo jurídico.


## ✅ Estructura del repositorio

```
innovaton_dj/
│
├─ juridica_model/
│   ├─ app.py                 # Interfaz Gradio (Lexi)
│   ├─ ingest.py              # Sincroniza Drive + indexa en Chroma
│   ├─ rag_chain.py           # Lógica RAG (búsqueda + generación)
│   ├─ drive_utils.py         # Funciones de descarga Drive
│   ├─ pdfs/                  # PDFs descargados   (git-ignored)
│   ├─ chroma_index/          # Base vectorial     (git-ignored)
│   ├─ static/
│   │   └─ Logotipo-CGR-blanco-transp.png
│   └─ requirements.txt
│
├─ .env                       # GEMINI_API_KEY, DRIVE_FOLDER_ID …
├─ service_account.json       # Credenciales de cuenta de servicio
└─ .gitignore

```

## ✅ Requisitos

```
| Herramienta       | Versión recomendada                                          |
| ----------------- | ------------------------------------------------------------ |
| Python            | 3.10 o superior                                              |
| Google Gemini API | Clave de Makersuite/AI Studio                                |
| Google Cloud      | Cuenta de servicio con acceso *read-only* a la carpeta Drive |
| Google Drive API  | Habilitada en el mismo proyecto                              |
| (Opc.) VS Code    | Para edición y virtualenv                                    |
```

## 📦 Instalación rápida

# 1. Clona el repo y entra
```
git clone <url> innovaton_dj
cd innovaton_dj/juridica_model
```
# 2. Crea y activa entorno virtual
```
python -m venv ../venv
../venv/Scripts/activate        # Windows
# source ../venv/bin/activate   # macOS / Linux
```
# 3. Instala dependencias
```
pip install -r requirements.txt
```
# 4. Variables de entorno (.env en la raíz)
```
GEMINI_API_KEY=tu_clave_API
DRIVE_FOLDER_ID=id_folder_drive_url
```
- Nota: service_account.json debe estar en juridica_model/ y la carpeta de Drive compartida con esa cuenta.

### 1. Uso diario

# 1. Descargar nuevos PDF + re-indexar
```
python ingest.py
```
# 2. Levantar la interfaz Lexi
```
python app.py
```

- Abre http://localhost:7860 y pregunta, por ejemplo:
```
¿Cuál es la sanción impuesta en el acto final N.º 07685-2025?
```
- Lexi mostrará la ficha completa (con la sanción “Separación del cargo público sin responsabilidad patronal”, etc.) y luego responderá preguntas de seguimiento sin repetir la ficha.

### 2. Archivos ignorados (.gitignore)


# Credenciales
```
.env
service_account.json
```
# Datos y artefactos
```
pdfs/
chroma_index/
```
# Entorno virtual
```
venv/
.venv/
```
# Byte-code
```
__pycache__/
*.py[cod]
```

### 3. Solución de problemas

```
| Error                               | Causa                                                     | Solución                                                               |
| ----------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `404 File not found` en `list_pdfs` | `DRIVE_FOLDER_ID` incorrecto o carpeta no compartida      | Verifique el ID y comparta la carpeta con la cuenta de servicio        |
| `429 ResourceExhausted`             | Se agotaron las 50 peticiones gratuitas diarias de Gemini | Espere al día siguiente, cambie a `gemini-pro`, o habilite facturación |
| Fuentes negras en el chat           | Tema oscuro sobreescribe estilos                          | El proyecto fuerza tema claro y CSS personalizados                     |

```

### 🧹 Limpieza opcional

- Si venías de usar transformers, torch, u otras dependencias de modelos locales, podés desinstalarlas así:
```
pip uninstall transformers peft accelerate bitsandbytes datasets \
               torch scikit-learn pandas numpy

```
### 💬 Créditos

- Desarrollado como sistema de apoyo a la División Jurídica para análisis de resoluciones en lenguaje natural, utilizando inteligencia artificial generativa.
