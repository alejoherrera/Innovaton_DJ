# juridica_model/drive_utils.py
from pathlib import Path
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = Path(__file__).with_name("service_account.json")

def _drive():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def list_pdfs(folder_id: str):
    """Devuelve una lista de diccionarios {id, name} para cada PDF."""
    results = _drive().files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id,name)").execute()
    return results.get("files", [])

def download_file(file_id: str, outfile: Path):
    """Descarga el PDF con file_id a la ruta outfile."""
    request = _drive().files().get_media(fileId=file_id)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(outfile, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return outfile
