# Notebook setup without `subprocess`

Use this in notebooks instead of Python `subprocess.run(...)`.

## Local / Nix / VS Codium Jupyter

Run the notebook from the project root. Do not clone inside itself.

```python
from pathlib import Path
import os, sys

PROJECT_MARKERS = ("pyproject.toml", "src/yolo_chess")

def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "yolo_chess").is_dir():
            return p
    raise RuntimeError("Project root not found. Open notebook from the repo root.")

REPO_DIR = find_project_root()
os.chdir(REPO_DIR)
SRC = REPO_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

print("Repo dir:", REPO_DIR)
print("Python:", sys.executable)
print("No dependency installation was run from the notebook.")
```

## Real Google Colab

Use IPython magics, not `subprocess`.

```python
from pathlib import Path
import os, sys, importlib.util, urllib.request, zipfile, shutil

REPO_URL_ZIP = "https://github.com/misssglory/yolo41/archive/refs/heads/main.zip"
REPO_DIR = Path("/content/yolo41")

if not REPO_DIR.exists():
    zip_path = Path("/content/yolo41-main.zip")
    urllib.request.urlretrieve(REPO_URL_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/content")
    extracted = Path("/content/yolo41-main")
    if not extracted.exists():
        extracted = next(Path("/content").glob("yolo41-*"))
    shutil.move(str(extracted), str(REPO_DIR))

os.chdir(REPO_DIR)
SRC = REPO_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

print("Repo dir:", REPO_DIR)
print("Python:", sys.executable)
```

Then run in a separate Colab cell:

```python
%pip install -q -r requirements-colab.txt
%pip install -q -e . --no-deps
```
