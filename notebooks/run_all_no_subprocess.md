# Run all from ipynb without subprocess

## Cell 0 — environment before TensorFlow import

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
```

## Cell 1 — local project setup, no clone, no subprocess

```python
from pathlib import Path
import os
import sys


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
```

## Cell 1b — real Colab only: install deps with IPython magic

Do not run this locally in Nix.

```python
%pip install -q -r requirements-colab.txt
%pip install -q -e . --no-deps
```

## Cell 2 — import TensorFlow and download dataset using Python, no subprocess

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

tf.config.optimizer.set_jit(False)
tf.config.set_soft_device_placement(True)

from yolo_chess.download import ensure_data_yaml

DATA_YAML = ensure_data_yaml("chess_yolo/data.yaml", download_if_missing=True)
CONFIG_PATH = "config.toml"

print("DATA_YAML:", DATA_YAML)
```

## Cell 3 — check class mapping and GT polygon labels inline

```python
from yolo_chess.gt_viz import (
    collect_images_from_split,
    show_single_ground_truth,
    show_ground_truth_grid,
)
from yolo_chess.config import load_config

cfg = load_config(CONFIG_PATH)
print("label box_format:", cfg.labels.box_format)

images, class_names, info = collect_images_from_split(DATA_YAML, split="valid", config_path=CONFIG_PATH)

print("Images:", len(images))
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i:2d} -> {name}")

show_single_ground_truth(images, class_names, config_path=CONFIG_PATH, draw_imgsz=640)
show_ground_truth_grid(images, class_names, config_path=CONFIG_PATH, sample_count=16, grid_cols=4, draw_imgsz=640)
```

## Cell 4 — train from notebook without subprocess

```python
from yolo_chess.notebook_api import train_notebook

result = train_notebook(
    config_path=CONFIG_PATH,
    data_yaml=DATA_YAML,
    epochs=15,
    batch=2,
    lr=1e-4,
    device="auto",
    weights="none",
    pretrained_darknet=True,
    out="runs/detect/yolov3_keras_chess",
    no_increment=False,
)

BEST_WEIGHTS = result["best_weights"]
CLASS_NAMES = result["dataset_info"].class_names
NUM_CLASSES = result["dataset_info"].num_classes

print("BEST_WEIGHTS:", BEST_WEIGHTS)
```

## Cell 5 — plot training loss inline

```python
import matplotlib.pyplot as plt

history = result["history"].history

plt.figure(figsize=(9, 5))
plt.plot(history.get("loss", []), label="train loss")
if "val_loss" in history:
    plt.plot(history["val_loss"], label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Cell 6 — inference inline on one image and grid

```python
from pathlib import Path
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from yolo_chess.infer import build_inference_model, predict_image_with_meta

model = build_inference_model(BEST_WEIGHTS, NUM_CLASSES)

test_images, _, _ = collect_images_from_split(DATA_YAML, split="test", config_path=CONFIG_PATH)
print("Test images:", len(test_images))

image_path = random.choice(test_images)
drawn_bgr, report = predict_image_with_meta(model, image_path, CLASS_NAMES, conf=0.35)
drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)

print("Image:", image_path)
print("Detections:", len(report["detections"]))
for det in report["detections"]:
    print(det["class_id"], det["class_name"], det["score"], det["box_restored_original_xyxy"])

plt.figure(figsize=(8, 8))
plt.imshow(drawn_rgb)
plt.axis("off")
plt.show()
```

## Cell 7 — prediction grid inline

```python
SAMPLE_COUNT = 16
GRID_COLS = 4
samples = random.sample(test_images, min(SAMPLE_COUNT, len(test_images)))
rows = math.ceil(len(samples) / GRID_COLS)

fig, axes = plt.subplots(rows, GRID_COLS, figsize=(18, 18))
axes = np.array(axes).reshape(-1)

for ax, img_path in zip(axes, samples):
    drawn_bgr, report = predict_image_with_meta(model, img_path, CLASS_NAMES, conf=0.35)
    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(drawn_rgb)
    ax.set_title(f"detections={len(report['detections'])}", fontsize=8)
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
```
