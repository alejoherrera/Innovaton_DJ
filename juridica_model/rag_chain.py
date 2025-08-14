import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Importamos la función para descargar desde Drive
from .drive_utils import download_file_from_drive

# --- Variables Globales y de Configuración ---

# Usaremos una variable global para guardar la cadena y no tener que recargarla.
CONVERSATIONAL_CHAIN = None

# ID del archivo PDF en Google Drive (reemplaza con el ID real de tu archivo)
PDF_FILE_ID = "ID_DE_TU_ARCHIVO_PDF_EN_GOOGLE_DRIVE" 
PDF_FILENAME = "documento_local.pdf"

# Expresión regular que app.py también importa
GOODBYE_RE = re.compile(r"\b(adiós|chao|hasta luego)\b", re.IGNORECASE)


# --- Funciones Principales ---

def get_llm():
    """Obtiene la instancia del modelo de lenguaje de Google (Gemini)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Error: No se encontró la variable de entorno GEMINI_API_KEY.")
    return GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

def get_vector_store(file_path):
    """Crea y retorna un vector store a partir de un archivo PDF."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(texts, embedding=embeddings)

def get_conversational_chain(vector_store):
    """Crea y retorna la cadena conversacional."""
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

def initialize_chain():
    """
    Función de inicialización que descarga el PDF, crea el vector store 
    y la cadena conversacional. Se ejecuta solo una vez.
    """
    global CONVERSATIONAL_CHAIN
    print("Iniciando la cadena conversacional...")
    
    # 1. Descargar el archivo desde Google Drive
    print(f"Descargando archivo desde Drive con ID: {PDF_FILE_ID}")
    local_pdf_path = download_file_from_drive(PDF_FILE_ID, PDF_FILENAME)
    
    if not local_pdf_path:
        raise Exception("No se pudo descargar el archivo PDF desde Google Drive.")
    
    # 2. Crear el vector store desde el archivo descargado
    print("Creando vector store...")
    vector_store = get_vector_store(local_pdf_path)
    
    # 3. Crear y guardar la cadena conversacional en la variable global
    print("Creando cadena conversacional...")
    CONVERSATIONAL_CHAIN = get_conversational_chain(vector_store)
    print("¡Inicialización completada!")

def answer(query: str, k: int = 10):
    """
    Esta es la función que tu app.py necesita.
    Recibe una consulta, la procesa y devuelve la respuesta.
    """
    global CONVERSATIONAL_CHAIN
    
    # Si la cadena no ha sido inicializada, lo hacemos ahora.
    if CONVERSATIONAL_CHAIN is None:
        initialize_chain()
        
    # Usamos la cadena para obtener una respuesta
    result = CONVERSATIONAL_CHAIN({"question": query})
    
    # Devolvemos la respuesta y un valor placeholder (tu código original esperaba dos valores)
    return result["answer"], None

# --- Bloque de ejecución principal ---
if __name__ == '__main__':
    # Este código es útil para probar el módulo directamente
    print("Módulo rag_chain.py cargado. Probando la función de respuesta...")
    try:
        # Hacemos una pregunta de prueba para forzar la inicialización
        test_query = "Hola"
        response, _ = answer(test_query)
        print(f"Pregunta de prueba: {test_query}")
        print(f"Respuesta de prueba: {response}")
    except Exception as e:
        print(f"Ocurrió un error durante la prueba: {e}")

