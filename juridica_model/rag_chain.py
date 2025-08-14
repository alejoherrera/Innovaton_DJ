# Se importa la librería 'os' para poder acceder a las variables de entorno del sistema
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_llm():
    """
    Obtiene la instancia del modelo de lenguaje de Google (Gemini).

    Esta función ha sido refactorizada para leer la API key de una variable 
    de entorno llamada 'GEMINI_API_KEY', en lugar de tenerla escrita en el código.
    Esto es una práctica de seguridad fundamental.
    """
    # 1. Se intenta leer la variable de entorno.
    #    os.environ.get() es la forma segura de hacerlo en Python.
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 2. Se añade una validación para asegurar que la clave fue encontrada.
    #    Si la aplicación se despliega sin configurar la variable, fallará con un error claro.
    if not api_key:
        raise ValueError("Error: No se encontró la variable de entorno GEMINI_API_KEY. "
                         "Asegúrate de configurarla en tu servicio de Cloud Run.")
    
    # 3. Se inicializa el modelo de lenguaje usando la clave obtenida del entorno.
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key) 
    return llm

def get_vector_store(file_path):
    """
    Crea y retorna un vector store (base de datos de vectores) a partir de un archivo PDF.
    """
    # Carga el documento PDF desde la ruta especificada
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    # Divide el texto de las páginas en fragmentos más pequeños (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)
    
    # Crea los embeddings (representaciones numéricas) para los fragmentos de texto
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Crea el vector store usando FAISS, indexando los textos y sus embeddings
    vector_store = FAISS.from_documents(texts, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    """
    Crea y retorna la cadena conversacional que une el modelo, la memoria y el retriever.
    """
    # Obtiene el modelo de lenguaje ya configurado
    llm = get_llm()
    
    # Configura la memoria para que el chatbot recuerde el historial de la conversación
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Crea la cadena de recuperación conversacional
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain

# Este bloque se usa para ejecutar código solo cuando el script se llama directamente.
# Es útil para pruebas, pero no se ejecuta cuando se importa desde otro módulo.
if __name__ == '__main__':
    print("Módulo rag_chain.py cargado.")
    print("Este módulo proporciona las funciones para crear una cadena conversacional con RAG.")
    # No se ejecuta ninguna lógica de negocio aquí para mantener el módulo reutilizable.
    pass
