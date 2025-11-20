from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to your downloaded service account key
SERVICE_ACCOUNT_FILE = "service-account.json"

# Minimal scopes (read-only)
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly"
]

# Replace this with your EVENTS folder ID (from the Drive URL)
FOLDER_ID = "11TpKl8skEaUC-CWBZkxJli79YX--tPsZ"

# Authenticate
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# Build Drive client
drive_service = build("drive", "v3", credentials=creds)

# First, let's check if we can access the folder itself (supports shared drives)
try:
    folder_info = drive_service.files().get(  # type: ignore
        fileId=FOLDER_ID,
        supportsAllDrives=True,
        fields="id, name, mimeType, parents"
    ).execute()
    print(f"Folder found: {folder_info['name']}")
    print(f"Folder ID: {folder_info['id']}")
except Exception as e:
    print(f"Error accessing folder: {e}")
    print("Make sure:")
    print("1. The folder ID is correct")
    print("2. The service account has been shared with this folder or the shared drive")
    exit(1)

# Query all files inside the folder
results = drive_service.files().list(  # type: ignore
    q=f"'{FOLDER_ID}' in parents and trashed = false",
    includeItemsFromAllDrives=True,
    supportsAllDrives=True,
    fields="files(id, name, mimeType)"
).execute()

items = results.get("files", [])

if not items:
    print("No files found.")
else:
    for item in items:
        print(f"{item['name']} ({item['id']}) - {item['mimeType']}")