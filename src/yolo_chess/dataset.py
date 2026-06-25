from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf
import yaml

from .config import ANCHORS, ANCHOR_MASKS, DEFAULT_CHESS_CLASS_NAMES, IMAGE_SUFFIXES, MAX_BOXES, SIZE, load_config
from .losses import transform_targets


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    train_images: list[Path]
    val_images: list[Path]
    test_images: list[Path]
    class_names: list[str]
    num_classes: int


def _resolve_yaml_path(root: Path, value: str | list | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[Path] = []
        for item in value:
            out.extend(_resolve_yaml_path(root, item))
        return out

    value = str(value)
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return [path]


def _collect_images(paths: Iterable[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(candidate)
    return sorted(set(images))


def _normalize_class_names(names, nc: int, config_path: str | os.PathLike = "config.toml") -> list[str]:
    """Return class names in numeric YOLO id order: 0, 1, 2, ...

    Ultralytics data.yaml may store names either as a list or as a mapping.
    When it is a mapping, keys can be strings, so plain sorted(names) would
    produce 0, 1, 10, 11, 12, 2... and break label mapping.
    """
    cfg = load_config(config_path)
    if cfg.labels.class_names_override:
        class_names = list(cfg.labels.class_names_override)
    elif isinstance(names, dict):
        class_names = [f"class_{i}" for i in range(nc)]
        for key, value in names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < nc:
                class_names[idx] = str(value)
    elif isinstance(names, list):
        class_names = [str(x) for x in names]
    else:
        class_names = []

    if nc == 13 and cfg.labels.use_default_chess_names_if_nc_13:
        # Use the known lesson mapping if yaml is missing/garbled/too short.
        if len(class_names) != 13 or any(not str(x).strip() for x in class_names):
            class_names = list(DEFAULT_CHESS_CLASS_NAMES)

    if not class_names:
        class_names = [f"class_{i}" for i in range(nc)]

    if len(class_names) < nc:
        class_names.extend(f"class_{i}" for i in range(len(class_names), nc))
    elif len(class_names) > nc:
        class_names = class_names[:nc]

    return class_names


def load_dataset_info(data_yaml: str | os.PathLike, config_path: str | os.PathLike = "config.toml") -> DatasetInfo:
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            "Download the dataset first:\n"
            "  python scripts/download_dataset.py\n"
            "or pass --download-if-missing to train.py.\n"
            "If the dataset is elsewhere, pass --data /path/to/data.yaml"
        )
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = data_yaml.parent.resolve()

    names = data.get("names", [])
    num_classes = int(data.get("nc", len(names) if isinstance(names, list) else len(names or [])))
    class_names = _normalize_class_names(names, num_classes, config_path=config_path)

    train_images = _collect_images(_resolve_yaml_path(root, data.get("train")))
    val_images = _collect_images(_resolve_yaml_path(root, data.get("val")))
    test_images = _collect_images(_resolve_yaml_path(root, data.get("test")))

    # The chess_yolo dataset in the lesson can be flat: chess_yolo/train/*.jpg.
    # If yaml uses old paths or empty val/test, make sensible fallbacks.
    if not train_images:
        train_images = _collect_images([root / "train"])
    if not val_images:
        val_images = _collect_images([root / "valid", root / "val"])
    if not test_images:
        test_images = _collect_images([root / "test"])

    if not val_images:
        val_images = test_images or train_images[: max(1, len(train_images) // 5)]
    if not test_images:
        test_images = val_images

    return DatasetInfo(root, train_images, val_images, test_images, class_names, num_classes)


def find_label_path(image_path: Path) -> Path:
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the most likely path even if it does not exist; reader will yield empty labels.
    return candidates[0]


def _convert_label_row_to_xyxy(parts: list[str], box_format: str) -> list[float] | None:
    """Convert one label row to [x1, y1, x2, y2, class], normalized.

    The lesson loss/target code expects xyxy-normalized boxes.

    Supported input formats:
    - polygon_normalized: class x1 y1 x2 y2 x3 y3 x4 y4
      The current chess_yolo dataset uses this 4-point polygon format.
      It is converted to an axis-aligned detection bbox using min/max.
    - xyxy_normalized: class x1 y1 x2 y2
    - yolo_xywh:       class x_center y_center width height
    """
    if len(parts) < 5:
        return None

    cls = int(float(parts[0]))
    fmt = box_format.strip().lower()

    if fmt in {"polygon_normalized", "polygon", "poly", "quadrilateral", "xyxyxyxy_normalized", "roboflow_polygon"}:
        if len(parts) < 9:
            # A polygon-configured dataset row without 4 points is invalid for this export.
            return None
        coords = [float(x) for x in parts[1:9]]
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    elif fmt in {"yolo_xywh", "xywh", "ultralytics_xywh"}:
        x_center, y_center, width, height = map(float, parts[1:5])
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
    elif fmt in {"xyxy_normalized", "xyxy", "corners", "pascal_normalized"}:
        x1, y1, x2, y2 = map(float, parts[1:5])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    else:
        raise ValueError(
            f"Unsupported [labels].box_format={box_format!r}. "
            "Use 'polygon_normalized', 'xyxy_normalized', or 'yolo_xywh'."
        )

    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2, float(cls)]

def read_yolo_label_file(
    label_path: str | os.PathLike,
    max_boxes: int = MAX_BOXES,
    config_path: str | os.PathLike = "config.toml",
) -> np.ndarray:
    """Read label txt and return padded [x1, y1, x2, y2, class] normalized boxes.

    The function name is kept for compatibility, but the actual input format is
    controlled by [labels].box_format in config.toml.
    """
    label_path = Path(label_path)
    cfg = load_config(config_path)
    box_format = cfg.labels.box_format

    boxes: list[list[float]] = []
    if label_path.exists():
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            converted = _convert_label_row_to_xyxy(parts, box_format)
            if converted is None:
                continue
            boxes.append(converted)

    out = np.zeros((max_boxes, 5), dtype=np.float32)
    if boxes:
        arr = np.asarray(boxes[:max_boxes], dtype=np.float32)
        out[: len(arr)] = arr
    return out


def _load_example(image_path_bytes, label_path_bytes, config_path_bytes):
    image_path = image_path_bytes.decode("utf-8")
    label_path = label_path_bytes.decode("utf-8")
    config_path = config_path_bytes.decode("utf-8")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # The task dataset is 416x416, but resize is kept for safety and lesson alignment.
    image_rgb = cv2.resize(image_rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0

    labels = read_yolo_label_file(label_path, MAX_BOXES, config_path=config_path)
    return image, labels


def make_dataset(
    image_paths: list[Path],
    batch_size: int,
    num_classes: int,
    shuffle: bool = True,
    config_path: str | os.PathLike = "config.toml",
) -> tf.data.Dataset:
    label_paths = [find_label_path(p) for p in image_paths]

    image_strs = [str(p) for p in image_paths]
    label_strs = [str(p) for p in label_paths]

    ds = tf.data.Dataset.from_tensor_slices((image_strs, label_strs))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_strs), 1024), reshuffle_each_iteration=True)

    def mapper(image_path, label_path):
        image, labels = tf.numpy_function(
            _load_example,
            [image_path, label_path, tf.constant(str(config_path))],
            [tf.float32, tf.float32],
        )
        image.set_shape((SIZE, SIZE, 3))
        labels.set_shape((MAX_BOXES, 5))
        y0, y1, y2 = transform_targets(tf.expand_dims(labels, 0), ANCHORS, ANCHOR_MASKS, num_classes)
        return image, (y0[0], y1[0], y2[0])

    ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
