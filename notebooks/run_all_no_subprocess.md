# YOLOv3 chess homework — notebook run, no subprocess

This notebook flow uses direct Python imports/functions only. No `subprocess.run(...)` is needed.

It includes:

- polygon labels: `class x1 y1 x2 y2 x3 y3 x4 y4` → detection bbox;
- Cyrillic-safe rendering for matplotlib and PIL predictions;
- square + horizontal + vertical orientation demonstration by **center crop**, not stretch.

## Cell 0 — environment, before TensorFlow

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
```

## Cell 1 — project root and Cyrillic font

```python
from pathlib import Path
import os
import sys


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "yolo_chess").is_dir():
            return p
    raise RuntimeError("Project root not found. Run the notebook from the repo root or cd into it first.")


REPO_DIR = find_project_root()
os.chdir(REPO_DIR)
SRC = REPO_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_chess.font_utils import configure_matplotlib_cyrillic, find_cyrillic_font_path
font = configure_matplotlib_cyrillic()

print("Repo dir:", REPO_DIR)
print("Python:", sys.executable)
print("Cyrillic font:", find_cyrillic_font_path())
```

## Cell 2 — TensorFlow and dataset

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

tf.config.optimizer.set_jit(False)
tf.config.set_soft_device_placement(True)

from yolo_chess.download import ensure_data_yaml
from yolo_chess.config import load_config

CONFIG_PATH = "config.toml"
DATA_YAML = ensure_data_yaml("chess_yolo/data.yaml", download_if_missing=True)
cfg = load_config(CONFIG_PATH)

print("DATA_YAML:", DATA_YAML)
print("label box_format:", cfg.labels.box_format)
print("demo orientation_mode:", cfg.demo.orientation_mode)
print("demo rectangular_transform:", cfg.demo.rectangular_transform)
```

## Cell 3 — ground truth check, inline

```python
from yolo_chess.gt_viz import collect_images_from_split, show_single_ground_truth, show_ground_truth_grid

valid_images, CLASS_NAMES, info = collect_images_from_split(DATA_YAML, split="valid", config_path=CONFIG_PATH)

print("Images:", len(valid_images))
print("Classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i:2d} -> {name}")

show_single_ground_truth(valid_images, CLASS_NAMES, config_path=CONFIG_PATH, draw_imgsz=640)
show_ground_truth_grid(valid_images, CLASS_NAMES, config_path=CONFIG_PATH, sample_count=16, grid_cols=4, draw_imgsz=640)
```

## Cell 4 — show horizontal/vertical crop inputs, inline

```python
import random
from yolo_chess.orientation_viz import show_orientation_crop_inputs

test_images, _, _ = collect_images_from_split(DATA_YAML, split="test", config_path=CONFIG_PATH)
base_image = random.choice(test_images)

print("Base image:", base_image)
show_orientation_crop_inputs(base_image, config_path=CONFIG_PATH, include_square=True)
```

## Cell 5 — remove old runs trained with wrong labels/LR

```python
import shutil
from pathlib import Path

runs_root = Path("runs/detect")
for p in runs_root.glob("yolov3_keras_chess*"):
    print("Removing:", p)
    shutil.rmtree(p)
print("Old runs removed.")
```

## Cell 6 — safe training

```python
from yolo_chess.notebook_api import train_notebook

result = train_notebook(
    config_path=CONFIG_PATH,
    data_yaml=DATA_YAML,
    epochs=20,
    batch=2,
    lr=1e-5,
    fine_tune_lr=3e-6,
    clipnorm=1.0,
    freeze_darknet_epochs=3,
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

## Cell 7 — loss plot

```python
import matplotlib.pyplot as plt

history = result["history"].history

plt.figure(figsize=(9, 5))
plt.plot(history.get("loss", []), label="train loss")
if "val_loss" in history:
    plt.plot(history["val_loss"], label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(prop=font)
plt.grid(True, alpha=0.3)
plt.show()
```

## Cell 8 — single prediction

```python
import random
import cv2
import matplotlib.pyplot as plt
from yolo_chess.infer import build_inference_model, predict_image_with_meta

model = build_inference_model(BEST_WEIGHTS, NUM_CLASSES)

image_path = random.choice(test_images)
drawn_bgr, report = predict_image_with_meta(model, image_path, CLASS_NAMES, conf=0.45)
drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)

print("Image:", image_path)
print("Detections:", len(report["detections"]))
for det in report["detections"]:
    print(det["class_id"], det["class_name"], f'{det["score"]:.3f}', det["box_restored_original_xyxy"])

plt.figure(figsize=(8, 8))
plt.imshow(drawn_rgb)
plt.axis("off")
plt.show()
```

## Cell 9 — prediction grid

```python
import math
import numpy as np

SAMPLE_COUNT = 16
GRID_COLS = 4
samples = random.sample(test_images, min(SAMPLE_COUNT, len(test_images)))
rows = math.ceil(len(samples) / GRID_COLS)

fig, axes = plt.subplots(rows, GRID_COLS, figsize=(18, 18))
axes = np.array(axes).reshape(-1)

for ax, img_path in zip(axes, samples):
    drawn_bgr, report = predict_image_with_meta(model, img_path, CLASS_NAMES, conf=0.45)
    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(drawn_rgb)
    ax.set_title(f"detections={len(report['detections'])}", fontproperties=font, fontsize=8)
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

## Cell 10 — square + horizontal + vertical prediction by crop windows

Rectangular images are shown as real center-crops, but inference is done with overlapping square crop windows.
This avoids the failure mode where full-image letterbox shrinks pieces too much on portrait/landscape inputs.

```python
from yolo_chess.orientation_viz import (
    show_orientation_predictions,
    show_orientation_letterbox_vs_crop_windows,
)

reports = show_orientation_predictions(
    model,
    base_image,
    CLASS_NAMES,
    config_path=CONFIG_PATH,
    conf=0.35,
    include_square=True,
    use_crop_windows=True,
    crop_overlap=0.35,
    nms_iou=0.45,
)

for r in reports:
    print(
        r["image_path"],
        "mode=", r.get("inference_mode"),
        "size=", r["original_size"],
        "detections=", len(r["detections"]),
    )
```

Optional debug comparison:

```python
comparison_reports = show_orientation_letterbox_vs_crop_windows(
    model,
    base_image,
    CLASS_NAMES,
    config_path=CONFIG_PATH,
    conf=0.35,
    crop_overlap=0.35,
    nms_iou=0.45,
)
```
