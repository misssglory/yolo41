# Multi-crop voting inference cells, no subprocess

## 1. Load latest `best.weights.h5` and build model

```python
from pathlib import Path
import os
import sys
import tensorflow as tf

CONFIG_PATH = "config.toml"
DATA_YAML = "chess_yolo/data.yaml"

# Find project root and attach src.
REPO_DIR = Path.cwd().resolve()
for p in [REPO_DIR, *REPO_DIR.parents]:
    if (p / "src" / "yolo_chess").is_dir():
        REPO_DIR = p
        break

os.chdir(REPO_DIR)
SRC = REPO_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Avoid XLA/NMS surprises in notebooks.
tf.config.optimizer.set_jit(False)
tf.config.set_soft_device_placement(True)

from yolo_chess.dataset import load_dataset_info
from yolo_chess.infer import build_inference_model

dataset_info = load_dataset_info(DATA_YAML, config_path=CONFIG_PATH)
CLASS_NAMES = dataset_info.class_names
NUM_CLASSES = dataset_info.num_classes

def find_latest_best_weights(runs_root="runs/detect"):
    candidates = list(Path(runs_root).glob("yolov3_keras_chess*/weights/best.weights.h5"))
    if not candidates:
        raise FileNotFoundError("No best.weights.h5 found. Train the model first.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

BEST_WEIGHTS = find_latest_best_weights()
model = build_inference_model(BEST_WEIGHTS, NUM_CLASSES)

print("REPO_DIR:", REPO_DIR)
print("BEST_WEIGHTS:", BEST_WEIGHTS)
print("NUM_CLASSES:", NUM_CLASSES)
print("model loaded")
```

## 2. Run multi-crop voting on one square test image

```python
import random
from yolo_chess.gt_viz import collect_images_from_split
from yolo_chess.multi_crop_voting import show_multicrop_voting_inline

test_images, _, _ = collect_images_from_split(DATA_YAML, split="test", config_path=CONFIG_PATH)
base_image = random.choice(test_images)

print("Image:", base_image)

report = show_multicrop_voting_inline(
    model,
    base_image,
    CLASS_NAMES,
    config_path=CONFIG_PATH,
    figsize=(9, 9),
)

print("raw detections before fusion:", report["detections_before_fusion_count"])
print("fused detections:", len(report["detections"]))

for det in report["detections"]:
    print(
        det["class_id"],
        det["class_name"],
        f"final={det['final_score']:.3f}",
        f"mean={det['mean_score']:.3f}",
        f"votes={det['votes']}/{det['support_denominator']}",
        det["box_restored_original_xyxy"],
    )
```

## 3. Debug individual crops before fusion

```python
from yolo_chess.multi_crop_voting import show_multicrop_debug_grid_inline

_ = show_multicrop_debug_grid_inline(
    model,
    base_image,
    CLASS_NAMES,
    config_path=CONFIG_PATH,
    max_crops=16,
    cols=4,
)
```

## 4. Tune config from notebook without editing `config.toml`

```python
import cv2
import matplotlib.pyplot as plt
from yolo_chess.multi_crop_voting import (
    MultiCropVotingConfig,
    predict_multicrop_voting_image,
)

cfg = MultiCropVotingConfig(
    crop_fractions=(1.0, 0.92, 0.84, 0.76),
    offset_mode="center_and_cardinal",  # or "grid3"
    base_conf=0.10,
    fusion_iou=0.35,
    min_votes=1,
    min_final_score=0.18,
    support_denominator="eligible",
    score_weight=0.55,
    support_weight=0.45,
    alpha_min=0.18,
    alpha_max=0.85,
    draw_crop_windows=False,
)

# Save temporary config-independent result.
drawn_bgr, report = predict_multicrop_voting_image(
    model,
    base_image,
    CLASS_NAMES,
    config=cfg,
)

drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(9, 9))
plt.imshow(drawn_rgb)
plt.axis("off")
plt.show()

for det in report["detections"]:
    print(det["class_name"], f"final={det['final_score']:.3f}", f"votes={det['votes']}/{det['support_denominator']}")
```

## 5. Make a 4x4 grid of multi-crop voting predictions

```python
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yolo_chess.multi_crop_voting import predict_multicrop_voting_image

SAMPLE_COUNT = 16
GRID_COLS = 4
samples = random.sample(test_images, min(SAMPLE_COUNT, len(test_images)))
rows = math.ceil(len(samples) / GRID_COLS)

fig, axes = plt.subplots(rows, GRID_COLS, figsize=(18, 18))
axes = np.asarray(axes).reshape(-1)

for ax, image_path in zip(axes, samples):
    drawn_bgr, report = predict_multicrop_voting_image(
        model,
        image_path,
        CLASS_NAMES,
        config_path=CONFIG_PATH,
    )
    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(drawn_rgb)
    ax.set_title(
        f"fused={len(report['detections'])}, raw={report['detections_before_fusion_count']}",
        fontsize=8,
    )
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
```
