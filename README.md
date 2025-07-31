# 🧠 Extractor de Resoluciones Jurídicas con Gemini

Este proyecto permite analizar resoluciones jurídicas en formato PDF y extraer información clave (número de resolución, persona investigada, motivo, etc.) utilizando la API de **Gemini Flash o Gemini Pro** de Google.

---

## ✅ Requisitos

- Python 3.10 o superior
- Cuenta en [https://makersuite.google.com/](https://makersuite.google.com/) para obtener una clave de API
- VS Code (opcional pero recomendado)

---

## 📦 Instalación y configuración

### 1. Crear el proyecto

```
juridica_model/
├── insumos/             # Carpeta con los archivos PDF
├── .env                 # Archivo con tu API Key
├── juridica_gemini.py   # Script principal
├── resumen_por_pdf.json # Salida generada
└── requirements.txt     # Dependencias del proyecto.
```

### 2. Crear entorno virtual

- Desde la terminal en la raíz del proyecto, crea o activa un ambiente virtal para la ejecución de código de python.
```
python -m venv venv
```
- Activa ese entorno con el siguiente código (Windows)
```
.\venv\Scripts\Activate
```
### 3. Crear un archivo .env, para almacenar las variables de entorno. 

- En este caso la API de Gemini. La clave se obtiene desde: https://makersuite.google.com/app/apikey
```
GEMINI_API_KEY=tu_clave_aquí
```
### 4. Instalar dependencias

- Crea un archivo requirements.txt con lo siguiente:

google-generativeai
python-dotenv
PyPDF2
tqdm

- Después instala esas dependencias con el siguiente comando:
```
pip install -r requirements.txt
```
- Seguidamente se instalan las dependencias:
```
pip install google-generativeai python-dotenv PyPDF2 tqdm
```
## 📦 Uso

### 1. Ejecución

- Ejecuta el proyecto con el siguiente comando:
```
python juridica_gemini.py
```
### 2. Información esperada

- Se espera que el proyecto genere un archivo en formato .json con el resumen del análisis en un archivo llamado:

resumen_por_pdf.json

- La estructura esperada es:
```
[
  {
    "file_name": "resolucion_01.pdf",
    "respuestas": {
      "numero_resolucion": "Nº 18915-2024",
      "numero_interno": "DJ-234",
      "fecha_completa": "12 de mayo de 2024",
      ...
    }
  }
]
```
### Preguntas

- El sistema extrae las siguientes:

- ¿Cuál es el número de la resolución final?

- ¿Cuál es el número interno (DJ)?

- ¿Cuál es el número del procedimiento administrativo?

- ¿En qué fecha se emitió la resolución?

- ¿Contra quién se siguió el procedimiento administrativo?

- ¿Cuál fue el motivo del procedimiento sancionador?

- ¿Cuál fue el resultado o la sanción impuesta?


### 🧹 Limpieza opcional

- Si venías de usar transformers, torch, u otras dependencias de modelos locales, podés desinstalarlas así:

pip uninstall transformers peft accelerate bitsandbytes datasets scikit-learn torch pandas numpy


### 💬 Créditos

- Desarrollado como sistema de apoyo a la División Jurídica para análisis de resoluciones en lenguaje natural, utilizando inteligencia artificial generativa.
