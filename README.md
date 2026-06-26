# YOLOv3 Chess Homework

Проект выровнен под урок `41.2. Обнаружение объектов. Архитектура YOLOv3`.

В отличие от предыдущей PyTorch-версии, здесь модель собрана на TensorFlow/Keras в структуре урока:

- `DBL`
- `ResUnit`
- `ResN`
- `Darknet`
- `YoloHead`
- `YoloHeadOutput`
- `yolo_boxes`
- `nonMaximumSuppression`
- `YoloLoss`
- `transform_targets`
- `transform_targets_for_output`

Модель создаётся с нуля и обучается на датасете шахматных фигур.

## 1. Окружение

```bash
nix develop
```

Флейк создаёт `.venv` через `uv` и пытается автоматически поставить зависимости:

```bash
uv pip install --python .venv/bin/python -r requirements.txt -e .
```

> В `requirements.txt` используется `tf-nightly`, потому что проект оставлен на Python 3.14. Если TensorFlow nightly не ставится под вашу платформу, проще переключить `flake.nix` на `python313` или `python312` и поставить зависимости из `requirements-stable-python313.txt`.

## 2. Скачать датасет

```bash
python scripts/download_dataset.py
```

После этого должна появиться папка:

```text
chess_yolo/
  data.yaml
  train/
  valid/ или val/
  test/
```

Код умеет читать как плоскую структуру `train/*.jpg + train/*.txt`, так и обычную YOLO-структуру `images/ + labels/`.

## 3. Обучение

Обычный запуск:

```bash
python -m yolo_chess.train \
  --data chess_yolo/data.yaml \
  --epochs 50 \
  --batch 8 \
  --device auto \
  --weights auto
```

Запуск строго с нуля:

```bash
python -m yolo_chess.train \
  --data chess_yolo/data.yaml \
  --epochs 50 \
  --batch 8 \
  --device auto \
  --weights none
```

Загрузка конкретных весов перед обучением:

```bash
python -m yolo_chess.train \
  --data chess_yolo/data.yaml \
  --epochs 20 \
  --batch 8 \
  --device auto \
  --weights runs/detect/yolov3_keras_chess/weights/last.weights.h5
```

Результаты сохраняются в:

```text
runs/detect/yolov3_keras_chess/
  weights/best.weights.h5
  weights/last.weights.h5
  weights/final.weights.h5
  loss.png
  history.csv
  keras_log.csv
  classes.txt
  training_summary.txt
```

## 4. Предсказание

```bash
python -m yolo_chess.predict \
  --weights runs/detect/yolov3_keras_chess/weights/best.weights.h5 \
  --data chess_yolo/data.yaml \
  --source chess_yolo/test \
  --out runs/detect/predict \
  --conf 0.25
```

Для каждого изображения сохраняются:

```text
runs/detect/predict/*_pred.jpg
runs/detect/predict/metadata/*.json
```

В JSON есть:

- исходный размер изображения;
- размер входа сети `416x416`;
- параметры letterbox: `scale`, `pad_x`, `pad_y`;
- bbox в координатах сети `box_net_416_xyxy`;
- bbox после восстановления в исходные координаты `box_restored_original_xyxy`.

## 5. Демонстрация требования задания

Задание требует показать работу на изображениях разного размера: квадратном, альбомном и портретном.

```bash
python -m yolo_chess.demo_shapes \
  --weights runs/detect/yolov3_keras_chess/weights/best.weights.h5 \
  --data chess_yolo/data.yaml \
  --source chess_yolo/test \
  --out runs/detect/demo_shapes \
  --conf 0.25
```

При `orientation_mode = "mixed"` скрипт создаёт три входа:

```text
runs/detect/demo_shapes/inputs/original_416_square.jpg
runs/detect/demo_shapes/inputs/landscape_800x500.jpg
runs/detect/demo_shapes/inputs/portrait_500x900.jpg
```

И три результата с сохранением исходного размера:

```text
runs/detect/demo_shapes/original_416_square_pred.jpg
runs/detect/demo_shapes/landscape_800x500_pred.jpg
runs/detect/demo_shapes/portrait_500x900_pred.jpg
```

Также создаётся отчёт:

```text
runs/detect/demo_shapes/restoration_report.csv
runs/detect/demo_shapes/metadata/*.json
```

Там явно видно:

```text
original_width/original_height
network_width/network_height = 416/416
output_width/output_height = исходный размер
scale
pad_x/pad_y
first_box_net_416_xyxy
first_box_restored_original_xyxy
```


## 5.1. Baseline / orientation config

Теперь режим демонстрации управляется через `config.toml`.

По умолчанию включён честный baseline — только квадратное изображение `416x416`, как в датасете:

```toml
[demo]
orientation_mode = "square"
rectangular_transform = "center_crop"
```

Для проверки требования задания на альбомной и портретной ориентации включите смешанный режим:

```toml
[demo]
orientation_mode = "mixed"
rectangular_transform = "center_crop"
```

`orientation_mode = "square"` создаёт только:

```text
original_416_square.jpg
```

`orientation_mode = "mixed"` создаёт:

```text
original_416_square.jpg
landscape_800x500.jpg
portrait_500x900.jpg
```

Важно: раньше demo-скрипт делал обычный `cv2.resize(img, (w, h))`, то есть растягивал квадратную картинку в прямоугольник. Это деформировало шахматные фигуры и могло давать ложные срабатывания. Теперь для прямоугольных тестов используется `center_crop`: изображение масштабируется с сохранением пропорций, заполняет целевой формат и затем обрезается по центру. Альтернатива — `rectangular_transform = "letterbox"`, если нужно сохранить всю картинку с паддингом.

Можно переопределить режим без правки файла:

```bash
python -m yolo_chess.demo_shapes \
  --weights runs/detect/yolov3_keras_chess/weights/best.weights.h5 \
  --data chess_yolo/data.yaml \
  --source chess_yolo/test \
  --out runs/detect/demo_shapes_square \
  --orientation-mode square

python -m yolo_chess.demo_shapes \
  --weights runs/detect/yolov3_keras_chess/weights/best.weights.h5 \
  --data chess_yolo/data.yaml \
  --source chess_yolo/test \
  --out runs/detect/demo_shapes_mixed \
  --orientation-mode mixed \
  --rectangular-transform center_crop
```

В результатах дополнительно сохраняется:

```text
runs/detect/demo_shapes/effective_config.json
```

Там видно, какой режим реально применился при генерации demo-изображений.

## 6. Как восстанавливаются bbox

Инференс делает не обычный resize, а letterbox:

```text
исходное изображение
→ масштабирование с сохранением пропорций
→ padding до 416x416
→ YOLOv3 prediction
→ bbox из координат 416x416 обратно в исходные координаты
```

Формула восстановления:

```python
x_original = (x_network - pad_x) / scale
y_original = (y_network - pad_y) / scale
```

После этого координаты ограничиваются размерами исходного изображения через `clip`.


## Fix: `FileNotFoundError: chess_yolo/data.yaml`

That error means the dataset has not been downloaded/unpacked in the project root yet.

Run:

```bash
python scripts/download_dataset.py
```

or simply run training again. `train.py` now downloads the default lesson dataset automatically when `chess_yolo/data.yaml` is missing:

```bash
python -m yolo_chess.train --data chess_yolo/data.yaml --epochs 5 --batch 8 --device auto --weights auto
```

Disable auto-download if needed:

```bash
python -m yolo_chess.train --data chess_yolo/data.yaml --no-download-if-missing
```

## 7. Colab / Jupyter dependency install without uv

Colab may have an `uv` binary in PATH, but it can still fail when asked to install into `/usr/bin/python3`. Control this via `config.toml`:

```toml
[environment]
use_uv = false
requirements = "requirements-colab.txt"
editable_install = true
upgrade_pip = false
```

Then in the first notebook cell after cloning the repo, run:

```python
import os, sys, subprocess
from pathlib import Path

REPO_DIR = Path("/content/yolov3_chess_homework_lesson/yolov3_chess_homework_lesson")
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "scripts/install_deps.py", "--config", "config.toml"], check=True)
```

If you want to override the config from a notebook cell:

```python
subprocess.run([sys.executable, "scripts/install_deps.py", "--force-pip"], check=True)
```

For Nix + Python 3.14 you can keep using `requirements.txt` with `tf-nightly`; for Colab/Python 3.12 prefer `requirements-colab.txt` with stable `tensorflow`.

## Label mapping and Cyrillic annotations

The project reads class names from `chess_yolo/data.yaml` in the same way as the YOLOv11 lesson workflow with Ultralytics: numeric label ids in `.txt` files are mapped to `names` from `data.yaml`.

For the lesson chess dataset the expected order is:

```text
0  слон
1  черный слон
2  черный король
3  черный конь
4  черная пешка
5  черный ферзь
6  черная ладья
7  белый слон
8  белый король
9  белый конь
10 белая пешка
11 белый ферзь
12 белая ладья
```

You can verify the mapping and per-class counts with:

```bash
python scripts/check_labels.py --data chess_yolo/data.yaml --config config.toml
```

Cyrillic labels are now drawn with PIL instead of `cv2.putText`, because OpenCV text rendering often shows Russian text as `????`.

The inference NMS also explicitly converts boxes from project format `[x1, y1, x2, y2]` to TensorFlow NMS format `[y1, x1, y2, x2]` and then converts them back before drawing. This fixes stretched/wrongly oriented boxes after `combined_non_max_suppression`.

## Robust notebook / Colab bootstrap

Use this first cell both in Colab and local Jupyter. It detects real Colab, avoids reinstalling TensorFlow in Colab, and falls back to `uv` in local Nix venvs that do not contain `pip`.

```python
from pathlib import Path
import os
import sys
import subprocess

REPO_URL = "https://github.com/misssglory/yolo41.git"
WORKDIR = Path.cwd()
REPO_DIR = WORKDIR / "yolov3_chess_homework_lesson"

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print(f"Repo already exists: {REPO_DIR}")
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))

print("Repo dir:", REPO_DIR)
print("Python:", sys.executable)

# Let the project decide: Colab -> pip without TensorFlow reinstall;
# local/Nix -> uv if pip is missing.
subprocess.run(
    [sys.executable, "scripts/install_deps.py", "--config", "config.toml"],
    check=True,
)
```

To inspect what it detected without installing anything:

```bash
python scripts/install_deps.py --config config.toml --print-env
```

## Notebook setup without subprocess

Do not clone the repository into itself from a local notebook. If `Path.cwd()` is already the project root, just add `src` to `sys.path`.
See `notebooks/setup_no_subprocess.md` for local and Colab cells that avoid `subprocess.run(...)`.

For local Nix / VS Codium, use `nix develop` first and run Jupyter inside that environment.
For real Colab, use `%pip` magics in a separate cell; do not run `python -m pip` through Python `subprocess`.

## Current label format fix: polygon labels

The chess dataset used in this homework stores labels as **4-point polygons**, not as standard YOLO `x_center y_center width height` rows.

Each `.txt` row is:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

All coordinates are normalized to `0..1`. For YOLOv3 detection training the project converts every polygon to a normal axis-aligned bbox:

```text
x_min = min(x1, x2, x3, x4)
y_min = min(y1, y2, y3, y4)
x_max = max(x1, x2, x3, x4)
y_max = max(y1, y2, y3, y4)
```

This is configured in `config.toml`:

```toml
[labels]
box_format = "polygon_normalized"
```

If you replace the dataset with a standard YOLO/Ultralytics export, change it to:

```toml
[labels]
box_format = "yolo_xywh"
```

Ground-truth visualization is available both as a script and notebook helpers:

```bash
python scripts/check_labels.py --data chess_yolo/data.yaml --config config.toml
python scripts/draw_ground_truth.py --data chess_yolo/data.yaml --config config.toml --split valid --imgsz 640
```

For notebooks without `subprocess`, use functions from:

```python
from yolo_chess.gt_viz import collect_images_from_split, show_single_ground_truth, show_ground_truth_grid
from yolo_chess.notebook_api import train_notebook
```

## Latest fixes: polygon labels, Cyrillic labels, and crop orientation demo

The chess label files in this dataset are polygons, not standard YOLO `xywh` boxes:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

The loader converts each polygon to the axis-aligned detection bbox required by the lesson YOLOv3 target format:

```text
bbox = [min(xs), min(ys), max(xs), max(ys)]
```

Cyrillic rendering is handled by `yolo_chess.font_utils`. It searches for DejaVu/Liberation fonts through common Linux/Nix/Colab paths and matplotlib's bundled fonts. If your environment still renders Russian labels incorrectly, set:

```bash
export YOLO_CHESS_FONT=/path/to/DejaVuSans.ttf
```

The no-subprocess notebook is here:

```text
notebooks/run_all_no_subprocess.ipynb
```

It includes an inline demonstration of:

- original square image;
- horizontal landscape image created by center crop;
- vertical portrait image created by center crop;
- prediction boxes restored to each image's own size.

Important inference detail: rectangular demo inputs are created by crop, but predictions on those rectangular images use overlapping square crop windows by default. Full-image letterbox inference shrinks objects inside a padded 416x416 input, which can remove detections on landscape/portrait images. Crop-window inference keeps object scale close to square training images and then maps boxes back into the landscape/portrait coordinate system.

Notebook functions:

```python
from yolo_chess.orientation_viz import (
    show_orientation_predictions,
    show_orientation_letterbox_vs_crop_windows,
)
```

This satisfies the homework requirement to demonstrate output restoration on images of different sizes and both landscape/portrait orientation without stretching the image.
