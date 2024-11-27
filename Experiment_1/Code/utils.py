import os

def ensure_directory_exists(directory):
    """
    Erstellt das Verzeichnis, falls es nicht existiert.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_message(message):
    """
    Protokolliert eine Nachricht auf der Konsole.
    """
    print(f"[INFO] {message}")
