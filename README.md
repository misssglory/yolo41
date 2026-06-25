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
