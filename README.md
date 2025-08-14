# 🧠 RAG | Resoluciones de Acto Final – División Jurídica CGR

Este proyecto convierte el repositorio de actos finales en un sistema RAG (Retrieval-Augmented Generation) que:

- **Sincroniza automáticamente** los PDF desde una carpeta oficial de Google Drive
- **Indexa su contenido en Qdrant Cloud** usando embeddings de Gemini  
- **Expone un chat Gradio ("Lexi")** con respuestas amistosas y precisas para el equipo jurídico
- **Detecta automáticamente documentos nuevos** sin reprocesar los existentes
- **Se despliega en Google Cloud Run** para acceso 24/7

## ✅ Estructura del repositorio

```
innovaton_dj/
│
├── app.py                    # Interfaz Gradio (Lexi) para Cloud Run
├── rag_chain.py              # Lógica RAG con Qdrant + detección automática
├── drive_utils.py            # Funciones de descarga desde Google Drive
├── requirements.txt          # Dependencias optimizadas para producción
├── .env.example             # Plantilla de variables de entorno
├── .gitignore               # Protección de credenciales
└── static/
    └── Logotipo-CGR-transp.png
```

## 🏗️ Arquitectura

```
Google Drive (PDFs) → Cloud Run → Qdrant Cloud → Gemini AI → Gradio UI
```

### Componentes principales:

- **Google Drive**: Almacén centralizado de PDFs de resoluciones
- **Qdrant Cloud**: Base de datos vectorial persistente y escalable
- **Google Gemini**: Generación de embeddings y respuestas
- **Google Cloud Run**: Hosting serverless con escalamiento automático
- **Gradio**: Interfaz de chat amigable para usuarios

## 📦 Requisitos

| Herramienta | Versión recomendada |
|-------------|-------------------|
| Python | 3.10 o superior |
| Google Gemini API | Clave de AI Studio |
| Qdrant Cloud | Cuenta y cluster activo |
| Google Cloud | Proyecto con Cloud Run habilitado |
| Google Drive API | Cuenta de servicio con permisos de lectura |

## 🚀 Despliegue en Cloud Run

### 1. Configuración de credenciales

#### Google Drive:
- Crea una cuenta de servicio en Google Cloud Console
- Descarga las credenciales JSON
- Comparte la carpeta de Drive con el email de la cuenta de servicio

#### Qdrant Cloud:
- Crea un cluster en [Qdrant Cloud](https://cloud.qdrant.io)
- Obtén la URL y API key del cluster

#### Google Gemini:
- Obtén tu API key desde [AI Studio](https://makersuite.google.com/app/apikey)

### 2. Variables de entorno en Cloud Run

#### Opción A: Desde la consola web
1. Ve a **Cloud Run** en Google Cloud Console
2. Selecciona tu servicio o haz clic en **"Crear servicio"**
3. En **"Editar e implementar nueva revisión"**
4. Ve a la pestaña **"Variables y secretos"**
5. Haz clic en **"+ Agregar variable"** para cada una:

```
Nombre: GEMINI_API_KEY
Valor: tu_gemini_api_key_aqui

Nombre: QDRANT_URL  
Valor: https://tu-cluster-id.us-east4-0.gcp.cloud.qdrant.io:6333

Nombre: QDRANT_API_KEY
Valor: tu_qdrant_api_key_aqui
```

#### Opción B: Desde gcloud CLI
```bash
gcloud run services update innovaton-dj \
  --update-env-vars="GEMINI_API_KEY=tu_key" \
  --region=europe-west1

gcloud run services update innovaton-dj \
  --update-env-vars="QDRANT_URL=https://tu-cluster.qdrant.io:6333" \
  --region=europe-west1

gcloud run services update innovaton-dj \
  --update-env-vars="QDRANT_API_KEY=tu_qdrant_key" \
  --region=europe-west1
```

#### Opción C: Durante el deployment inicial
```bash
gcloud run deploy innovaton-dj \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --update-env-vars="GEMINI_API_KEY=tu_key,QDRANT_URL=tu_url,QDRANT_API_KEY=tu_qdrant_key"
```

### 3. Deployment

#### Opción A: Desde la consola web
1. Ve a **Cloud Run** → **"Crear servicio"**
2. Selecciona **"Implementar una revisión desde un repositorio existente"**
3. Conecta tu repositorio de GitHub
4. Configura las variables de entorno (ver sección anterior)
5. Establece **timeout: 900 segundos** para la primera inicialización
6. Haz clic en **"Crear"**

#### Opción B: Desde gcloud CLI
```bash
# Clonar el repositorio
git clone https://github.com/alejoherrera/Innovaton_DJ.git
cd Innovaton_DJ

# Desplegar en Cloud Run
gcloud run deploy innovaton-dj \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --timeout=900 \
  --update-env-vars="GEMINI_API_KEY=tu_key,QDRANT_URL=tu_url,QDRANT_API_KEY=tu_qdrant_key"
```

#### Configuración recomendada:
- **Región**: `europe-west1` (o tu región preferida)
- **CPU**: 1 vCPU mínimo
- **Memoria**: 2 GiB mínimo  
- **Timeout**: 900 segundos (15 minutos)
- **Concurrencia**: 80 solicitudes por instancia

## 🔄 Gestión de documentos

### Procesamiento automático:
- **Primera ejecución**: Procesa todos los PDFs de la carpeta de Drive
- **Reinicios posteriores**: Solo procesa archivos nuevos automáticamente
- **Archivos corruptos**: Se saltan automáticamente sin afectar el sistema

### Agregar nuevos documentos:
1. **Sube PDFs nuevos** a la carpeta de Google Drive
2. **Reinicia la aplicación** en Cloud Run (o espera al próximo reinicio automático)
3. **El sistema detecta y procesa** solo los archivos nuevos

### Forzar actualización completa:
- Elimina la colección en el dashboard de Qdrant Cloud
- Reinicia la aplicación para reprocesar todo desde cero

## 💬 Uso de la interfaz

### Ejemplos de consultas:
```
¿Cuál es la sanción impuesta en el acto final N.º 07685-2025?
```
```
Muéstrame resoluciones de 2024 con despido sin responsabilidad
```
```
Lista de expedientes con inhabilitación
```

### Comandos especiales:
- **Saludos**: "Hola", "Buenos días" → Mensaje de bienvenida
- **Despedidas**: "Adiós", "Gracias" → Mensaje de despedida
- **Cortesía**: "Gracias", "Perfecto" → Confirmación amable

## 🛠️ Desarrollo local

```bash
# 1. Clonar y configurar
git clone https://github.com/alejoherrera/Innovaton_DJ.git
cd Innovaton_DJ

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables (crear .env)
cp .env.example .env
# Editar .env con tus credenciales

# 5. Ejecutar localmente
python app.py
```

## 🔧 Solución de problemas

| Error | Causa | Solución |
|-------|-------|----------|
| `404 File not found` | Carpeta de Drive no accesible | Verificar ID de carpeta y permisos de cuenta de servicio |
| `Cannot read an empty file` | PDFs corruptos en Drive | El sistema los salta automáticamente |
| `429 ResourceExhausted` | Cuota de Gemini agotada | Esperar reset diario o habilitar facturación |
| Variables de entorno faltantes | Configuración incompleta | Verificar todas las variables en Cloud Run |
| Timeout en Cloud Run | Primera inicialización lenta | Aumentar timeout a 900 segundos |

## 🔒 Seguridad

- **Credenciales protegidas**: No se almacenan en el código
- **Variables de entorno**: Gestionadas por Cloud Run
- **Acceso de solo lectura**: La cuenta de servicio solo lee Drive
- **HTTPS**: Comunicación encriptada por defecto en Cloud Run

## 📊 Características técnicas

### Performance:
- **Primera carga**: 5-10 minutos (procesa todos los PDFs)
- **Reinicios**: < 30 segundos (verifica archivos nuevos)
- **Consultas**: < 3 segundos (búsqueda vectorial)

### Escalabilidad:
- **Documentos**: Soporta miles de PDFs
- **Usuarios concurrentes**: Escalamiento automático en Cloud Run
- **Almacenamiento**: Persistente en Qdrant Cloud

### Tolerancia a fallos:
- **Archivos corruptos**: Se saltan automáticamente
- **Errores de red**: Reintentos automáticos
- **Fallos de servicio**: Recuperación automática

## 💡 Roadmap futuro

- [ ] **Notificaciones automáticas** cuando se agregan documentos nuevos
- [ ] **Búsqueda por rangos de fechas** específicos  
- [ ] **Exportación de resultados** en PDF/Excel
- [ ] **Dashboard de analytics** para el uso del sistema
- [ ] **API REST** para integración con otros sistemas

## 💬 Créditos

Desarrollado como sistema de apoyo a la División Jurídica de la CGR para análisis de resoluciones en lenguaje natural, utilizando:

- **Qdrant Cloud** para almacenamiento vectorial
- **Google Gemini** para IA generativa
- **Google Cloud Run** para hosting serverless
- **Gradio** para interfaz de usuario
- **LangChain** para procesamiento de documentos

---

📧 **Contacto**: Para soporte técnico o mejoras, contactar al equipo de desarrollo.
- Desarrollado como sistema de apoyo a la División Jurídica para análisis de resoluciones en lenguaje natural, utilizando inteligencia artificial generativa.
