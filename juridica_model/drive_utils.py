# drive_utils.py (Adaptado para Cloud Run)
import io
import os
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Define los permisos que la aplicación necesita.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _get_drive_service():
    """Función helper para autenticar y crear el servicio de Drive."""
    try:
        # Usa las credenciales del entorno de Cloud Run (la cuenta de servicio)
        creds, _ = google.auth.default(scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"Error fatal durante la autenticación con Google Drive: {e}")
        return None

def list_pdf_files_in_folder(folder_id: str) -> list:
    """Lista todos los archivos PDF que se encuentran en una carpeta de Google Drive."""
    service = _get_drive_service()
    if not service:
        return []

    try:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=100, # Puedes ajustar este número si tienes más de 100 PDFs
            fields="nextPageToken, files(id, name)"
        ).execute()
        
        items = results.get('files', [])
        print(f"Se encontraron {len(items)} archivos PDF en la carpeta de Drive.")
        return items
    except HttpError as error:
        print(f"Ocurrió un error al listar los archivos de Drive: {error}")
        return []

def download_file_from_drive(file_id: str, output_filename: str) -> str | None:
    """Descarga un archivo específico de Google Drive usando su ID."""
    service = _get_drive_service()
    if not service:
        return None

    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Descargando '{output_filename}'... {int(status.progress() * 100)}%")

        with open(output_filename, "wb") as f:
            f.write(fh.getvalue())
            
        print(f"Archivo '{output_filename}' descargado exitosamente.")
        return output_filename
        
    except HttpError as error:
        print(f"Ocurrió un error al descargar el archivo {file_id}: {error}")
        return None
