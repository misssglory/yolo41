from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .config import IMAGE_SUFFIXES, load_config
from .dataset import DatasetInfo, find_label_path, load_dataset_info, read_yolo_label_file


@dataclass(frozen=True)
class RawLabelObject:
    class_id: int
    raw: str
    points_norm: list[tuple[float, float]] | None
    bbox_norm: tuple[float, float, float, float]


def collect_images_from_split(
    data_yaml: str | Path = "chess_yolo/data.yaml",
    split: str = "valid",
    config_path: str | Path = "config.toml",
) -> tuple[list[Path], list[str], DatasetInfo]:
    """Return image paths and class names for one split.

    split accepts: train, val/valid, test.
    """
    info = load_dataset_info(data_yaml, config_path=config_path)
    split_norm = split.strip().lower()
    if split_norm == "train":
        images = info.train_images
    elif split_norm in {"val", "valid", "validation"}:
        images = info.val_images
    elif split_norm == "test":
        images = info.test_images
    else:
        raise ValueError("split must be train, valid/val, or test")
    return images, info.class_names, info


def read_raw_label_objects(
    label_path: str | Path,
    config_path: str | Path = "config.toml",
) -> list[RawLabelObject]:
    """Read raw labels for visualization.

    For polygon_normalized rows, keeps the original 4 points for green polygon
    drawing and also computes the red detection bbox used by YOLOv3 training.
    For xyxy/yolo_xywh rows, points_norm is None and only bbox is drawn.
    """
    label_path = Path(label_path)
    cfg = load_config(config_path)
    fmt = cfg.labels.box_format.strip().lower()
    objects: list[RawLabelObject] = []
    if not label_path.exists():
        return objects

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        points: list[tuple[float, float]] | None = None

        if fmt in {"polygon_normalized", "polygon", "poly", "quadrilateral", "xyxyxyxy_normalized", "roboflow_polygon"}:
            if len(parts) < 9:
                continue
            coords = [float(x) for x in parts[1:9]]
            points = [
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            ]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        elif fmt in {"xyxy_normalized", "xyxy", "corners", "pascal_normalized"}:
            x1, y1, x2, y2 = map(float, parts[1:5])
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
        elif fmt in {"yolo_xywh", "xywh", "ultralytics_xywh"}:
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2
        else:
            raise ValueError(f"Unsupported [labels].box_format={cfg.labels.box_format!r}")

        x1 = max(0.0, min(1.0, float(x1)))
        y1 = max(0.0, min(1.0, float(y1)))
        x2 = max(0.0, min(1.0, float(x2)))
        y2 = max(0.0, min(1.0, float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append(RawLabelObject(class_id, raw, points, (x1, y1, x2, y2)))
    return objects


def _scale_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return point[0] * width, point[1] * height


def _scale_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 * width, y1 * height, x2 * width, y2 * height


def draw_ground_truth_matplotlib(
    image_path: str | Path,
    class_names: list[str],
    config_path: str | Path = "config.toml",
    draw_imgsz: int | None = 640,
    ax=None,
    show_title: bool = True,
    draw_polygon: bool = True,
    draw_bbox: bool = True,
    fontsize: int = 7,
    linewidth: int = 2,
):
    """Draw GT labels inline with matplotlib.

    Green polygon = original 4-point label if available.
    Red dashed box = axis-aligned bbox used by YOLOv3 detection training.
    """
    image_path = Path(image_path)
    label_path = find_label_path(image_path)

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    if draw_imgsz is not None:
        img = img.resize((int(draw_imgsz), int(draw_imgsz)), Image.BICUBIC)
    width, height = img.size

    objects = read_raw_label_objects(label_path, config_path=config_path)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.imshow(img)
    ax.axis("off")
    if show_title:
        ax.set_title(
            f"{image_path.name}\noriginal={original_size}, drawn={img.size}, objects={len(objects)}",
            fontsize=10,
        )

    for obj in objects:
        class_id = obj.class_id
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"BAD_CLASS_{class_id}"
        x1, y1, x2, y2 = _scale_bbox(obj.bbox_norm, width, height)

        if draw_polygon and obj.points_norm:
            points = [_scale_point(p, width, height) for p in obj.points_norm]
            poly = patches.Polygon(points, closed=True, linewidth=linewidth, edgecolor="lime", facecolor="none")
            ax.add_patch(poly)

        if draw_bbox:
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=linewidth,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

        ax.text(
            x1,
            max(0, y1 - 4),
            f"{class_id}: {class_name}",
            fontsize=fontsize,
            color="white",
            va="bottom",
            ha="left",
            bbox=dict(facecolor="red", alpha=0.85, edgecolor="none", pad=2),
        )

    return fig, ax, objects, label_path


def show_single_ground_truth(
    images: list[Path],
    class_names: list[str],
    image_path: str | Path | None = None,
    config_path: str | Path = "config.toml",
    draw_imgsz: int | None = 640,
):
    if image_path is None:
        image_path = random.choice(images)
    fig, ax, objects, label_path = draw_ground_truth_matplotlib(
        image_path,
        class_names,
        config_path=config_path,
        draw_imgsz=draw_imgsz,
        show_title=True,
        fontsize=9,
    )
    plt.show()
    print("Image:", image_path)
    print("Label:", label_path)
    print("Same stem:", Path(image_path).stem == label_path.stem if label_path else None)
    print("Objects:", len(objects))
    for obj in objects:
        name = class_names[obj.class_id] if 0 <= obj.class_id < len(class_names) else f"BAD_CLASS_{obj.class_id}"
        print(f"{obj.class_id:2d} -> {name:20s} raw: {obj.raw}")
    return objects


def show_ground_truth_grid(
    images: list[Path],
    class_names: list[str],
    config_path: str | Path = "config.toml",
    sample_count: int = 16,
    grid_cols: int = 4,
    draw_imgsz: int | None = 640,
    seed: int = 42,
):
    random.seed(seed)
    samples = random.sample(images, min(sample_count, len(images)))
    rows = math.ceil(len(samples) / grid_cols)
    fig, axes = plt.subplots(rows, grid_cols, figsize=(18, 18))
    axes = np.array(axes).reshape(-1)
    for ax, img_path in zip(axes, samples):
        draw_ground_truth_matplotlib(
            img_path,
            class_names,
            config_path=config_path,
            draw_imgsz=draw_imgsz,
            ax=ax,
            show_title=False,
            fontsize=6,
        )
    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return samples
