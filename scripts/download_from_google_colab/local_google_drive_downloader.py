import os
import sys
from glob import glob
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def in_google_colab():
    return "google.colab" in sys.modules


def find_project_root(current_dir=None):
    current_dir = current_dir or os.path.abspath(os.path.dirname(__file__))
    project_root_markers = ['.git', 'README.md', 'pyproject.toml']
    while True:
        if any(os.path.exists(os.path.join(current_dir, marker)) for marker in project_root_markers):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Project root could not be determined.")
        current_dir = parent_dir


def enforce_project_root_restriction(path):
    project_root = find_project_root()
    absolute_path = os.path.abspath(path)
    if not absolute_path.startswith(project_root):
        raise ValueError(f"Path '{absolute_path}' is outside the project root: '{project_root}'")
    return absolute_path


def find_credentials_file(credentials_dir, specified_file=None):
    credentials_dir = enforce_project_root_restriction(credentials_dir)
    if not os.path.exists(credentials_dir):
        raise FileNotFoundError(f"Credentials directory '{credentials_dir}' does not exist.")
    if specified_file:
        return specified_file
    json_files = glob(os.path.join(credentials_dir, "*.json"))
    if json_files:
        return json_files[0]
    raise FileNotFoundError("No JSON file found in the specified directory.")


def initialize_credentials(credentials_dir):
    credentials_file = find_credentials_file(credentials_dir)
    token_path = os.path.join(credentials_dir, "token.json")
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return creds


def setup_drive_service(credentials_dir_pattern):
    project_root = find_project_root()
    credentials_dir = enforce_project_root_restriction(os.path.join(project_root, credentials_dir_pattern))
    creds = initialize_credentials(credentials_dir)
    service = build("drive", "v3", credentials=creds)
    return service


def setup_drive_files(service):
    google_drive_project_base_path = "root/vis/NetworkSynth"
    data_path = "data/input"
    google_drive_project_path = os.path.join(google_drive_project_base_path, data_path)
    target_subfolders = {
        "Original Graphs": "Original Graphs",
        "position": "position",
        "sparse_matrices": "sparse_matrices",
    }

    data_folder_id = traverse_google_drive_path(service, google_drive_project_path)
    if not data_folder_id:
        print(f"Folder path '{google_drive_project_path}' not found in Google Drive.")
        return

    for subfolder_name, subfolder_local_name in target_subfolders.items():
        subfolder_id = find_drive_folder(service, subfolder_name, data_folder_id)
        if subfolder_id:
            local_path = enforce_project_root_restriction(
                os.path.join(find_project_root(), "data", "input", subfolder_local_name)
            )
            os.makedirs(local_path, exist_ok=True)
            download_folder_contents(service, subfolder_id, local_path)
        else:
            print(f"Subfolder '{subfolder_name}' not found under '{google_drive_project_path}'.")


def traverse_google_drive_path(service, full_path):
    path_components = full_path.split(os.path.sep)
    current_id = path_components[0]  # Start at "root"
    queue = path_components[1:]  # Remaining components to traverse

    while queue:
        folder_name = queue.pop(0)
        current_id = find_drive_folder(service, folder_name, current_id)
        if not current_id:
            return None  # Path component not found
    return current_id


def find_drive_folder(service, folder_name, parent_id="root"):
    results = service.files().list(
        q=f"'{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    for item in results.get("files", []):
        if item["name"] == folder_name:
            return item["id"]
    return None


def download_folder_contents(service, folder_id, local_path):
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    for item in results.get("files", []):
        file_id = item["id"]
        file_name = item["name"]
        file_path = os.path.join(local_path, file_name)
        print(f"Downloading {file_name} to {file_path}...")
        request = service.files().get_media(fileId=file_id)
        with open(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {file_name}: {int(status.progress() * 100)}% complete.")


def main():
    if in_google_colab():
        print("Running in Google Colab, simply mount Google Drive and access files.")
        return

    try:
        service = setup_drive_service("credentials")
        setup_drive_files(service)
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except HttpError as e:
        print(f"HttpError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
