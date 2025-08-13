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

# drive_utils.py
def list_pdfs(folder_id: str):
    """Devuelve una lista de {id, name} para cada PDF en Mi unidad o Drive compartido."""
    svc = _drive().files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false",
        fields="files(id,name)",
        includeItemsFromAllDrives=True,   # 🔑 ver items en drives compartidos
        supportsAllDrives=True,           # 🔑 habilitar soporte de drives compartidos
        corpora="allDrives"               # 🔑 buscar en todos los “drives”
    ).execute()
    return svc.get("files", [])

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
