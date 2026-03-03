import os
import urllib.request


def ensure_model(path: str, url: str) -> None:
    if not os.path.exists(path):
        print(f"Downloading model to {path}...")
        urllib.request.urlretrieve(url, path)
