# Se importan las librerías necesarias de Google y estándar de Python
import os
import io
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Define los permisos (scopes) que la aplicación necesita. 
# En este caso, solo necesita permiso para leer archivos de Drive.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def authenticate_google_drive():
    """
    Autentica con la API de Google Drive usando 'Application Default Credentials'.
    
    En un entorno como Cloud Run, esto utiliza automáticamente la Cuenta de Servicio
    que se le asigna al servicio, lo cual es seguro y no requiere archivos de credenciales.
    """
    try:
        # Esta es la línea clave: busca credenciales en el entorno de ejecución.
        creds, _ = google.auth.default(scopes=SCOPES)
        return creds
    except Exception as e:
        # Si falla la autenticación, se imprime un error claro.
        print(f"Error fatal durante la autenticación automática con Google: {e}")
        return None

def download_file_from_drive(file_id, output_filename):
    """
    Descarga un archivo específico de Google Drive usando su ID.
    
    Args:
        file_id (str): El ID único del archivo en Google Drive.
        output_filename (str): El nombre con el que se guardará el archivo en el 
                               contenedor de Cloud Run (ej: "documento.pdf").
        
    Returns:
        str: La ruta al archivo descargado si tiene éxito, o None si ocurre un error.
    """
    print(f"Iniciando descarga para el archivo con ID: {file_id}")
    
    # Primero, se obtienen las credenciales.
    creds = authenticate_google_drive()
    
    if not creds:
        print("Fallo en la autenticación. No se puede continuar con la descarga.")
        return None

    try:
        # Construye el cliente de la API de Drive para interactuar con el servicio.
        service = build("drive", "v3", credentials=creds)
        
        # Prepara la solicitud para obtener el contenido multimedia del archivo.
        request = service.files().get_media(fileId=file_id)
        
        # Se utiliza un buffer en memoria (BytesIO) para recibir los datos del archivo.
        fh = io.BytesIO()
        
        # Se inicializa el objeto que gestionará la descarga por fragmentos (chunks).
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Progreso de la descarga: {int(status.progress() * 100)}%")

        # Una vez completada la descarga, se escribe el contenido del buffer a un archivo.
        with open(output_filename, "wb") as f:
            f.write(fh.getvalue())
            
        print(f"El archivo '{output_filename}' se ha descargado exitosamente.")
        return output_filename
        
    except HttpError as error:
        # Manejo de errores específicos de la API de Google.
        print(f"Ocurrió un error con la API de Google Drive: {error}")
        return None
    except Exception as e:
        # Manejo de cualquier otro error inesperado.
        print(f"Ocurrió un error inesperado durante la descarga: {e}")
        return None

# Este bloque es para pruebas y no se ejecuta cuando el módulo es importado.
if __name__ == '__main__':
    print("Módulo drive_utils.py cargado.")
    print("Este módulo proporciona funciones para interactuar con Google Drive de forma segura.")
    pass
