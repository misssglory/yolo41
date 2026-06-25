from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

CHESS_DATASET_URL = "https://storage.yandexcloud.net/academy.ai/CV/chess_yolo.zip"
DEFAULT_DATA_YAML = Path("chess_yolo/data.yaml")
DEFAULT_ZIP = Path("chess_yolo.zip")
DEFAULT_DATASET_DIR = Path("chess_yolo")


def download_chess_dataset(
    url: str = CHESS_DATASET_URL,
    zip_path: str | Path = DEFAULT_ZIP,
    extract_to: str | Path = ".",
) -> Path:
    """Download and unpack the chess YOLO dataset used in the lesson.

    The lesson notebook does the same thing with shell commands:
    wget chess_yolo.zip && unzip chess_yolo.zip.
    This Python version is used by scripts so training can helpfully recover
    when chess_yolo/data.yaml is missing.
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    data_yaml = extract_to / DEFAULT_DATA_YAML

    if data_yaml.exists():
        print(f"Dataset already exists: {data_yaml}")
        return data_yaml

    if not zip_path.exists():
        print(f"Downloading chess dataset: {url}")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"Using existing archive: {zip_path}")

    print(f"Unpacking {zip_path} -> {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    except zipfile.BadZipFile:
        # Fallback for unusual archive detection; usually not needed.
        shutil.unpack_archive(str(zip_path), str(extract_to))

    if not data_yaml.exists():
        raise FileNotFoundError(
            "Dataset archive was unpacked, but chess_yolo/data.yaml was not found.\n"
            f"Expected: {data_yaml.resolve()}\n"
            "Check archive contents or pass --data /path/to/data.yaml explicitly."
        )

    print(f"Dataset ready: {data_yaml}")
    return data_yaml


def ensure_data_yaml(data_yaml: str | Path, download_if_missing: bool = True) -> Path:
    """Return an existing data.yaml path, optionally downloading default dataset.

    If user requested a custom data.yaml, we do not silently download over it;
    we raise a clear error with the exact command to run.
    """
    data_yaml = Path(data_yaml)
    if data_yaml.exists():
        return data_yaml

    default_like = data_yaml.as_posix() in {
        "chess_yolo/data.yaml",
        "./chess_yolo/data.yaml",
        str(DEFAULT_DATA_YAML),
    }

    if download_if_missing and default_like:
        print(f"data.yaml not found: {data_yaml}")
        print("Trying to download the lesson chess dataset automatically...")
        return download_chess_dataset()

    raise FileNotFoundError(
        f"data.yaml not found: {data_yaml}\n\n"
        "The dataset is not in the current project directory yet.\n"
        "Run one of these commands from the project root:\n"
        "  python scripts/download_dataset.py\n"
        "or:\n"
        "  python -m yolo_chess.train --data chess_yolo/data.yaml --download-if-missing\n\n"
        "If your dataset is elsewhere, pass the real path:\n"
        "  python -m yolo_chess.train --data /absolute/path/to/data.yaml"
    )
