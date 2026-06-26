from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import DemoConfig, load_config, orientation_cases
from .font_utils import configure_matplotlib_cyrillic, get_matplotlib_cyrillic_font
from .infer import predict_image_with_meta


def resize_cover_center_crop(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize without distortion: cover target rectangle and crop center.

    This creates true landscape/portrait test inputs while preserving chess-piece
    proportions. It is the requested crop mode, not stretching.
    """
    src_h, src_w = img.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    cropped = resized[top : top + target_h, left : left + target_w]

    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def make_crop_orientation_inputs(
    image_path: str | Path,
    config_path: str | Path = "config.toml",
    include_square: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Return [(case_name, BGR image)] for square, landscape and portrait crops."""
    cfg = load_config(config_path)
    demo = cfg.demo
    cases = [
        ("landscape_crop", demo.landscape_size),
        ("portrait_crop", demo.portrait_size),
    ]
    if include_square:
        cases = [("square_baseline", demo.square_size)] + cases

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out: list[tuple[str, np.ndarray]] = []
    for name, (w, h) in cases:
        crop = resize_cover_center_crop(img, int(w), int(h))
        out.append((f"{name}_{int(w)}x{int(h)}", crop))
    return out


def save_orientation_inputs(
    image_path: str | Path,
    out_dir: str | Path = "notebook_outputs/orientation_crop_inputs",
    config_path: str | Path = "config.toml",
    include_square: bool = True,
) -> list[tuple[str, Path]]:
    """Create crop-orientation files for inference functions that accept paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[tuple[str, Path]] = []
    stem = Path(image_path).stem
    for name, bgr in make_crop_orientation_inputs(image_path, config_path=config_path, include_square=include_square):
        path = out_dir / f"{stem}_{name}.jpg"
        cv2.imwrite(str(path), bgr)
        result.append((name, path))
    return result


def show_orientation_crop_inputs(
    image_path: str | Path,
    config_path: str | Path = "config.toml",
    include_square: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Show the actual square/landscape/portrait crop inputs inline."""
    font = configure_matplotlib_cyrillic()
    inputs = make_crop_orientation_inputs(image_path, config_path=config_path, include_square=include_square)
    cols = len(inputs)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    axes = np.array(axes).reshape(-1)
    for ax, (name, bgr) in zip(axes, inputs):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{name}\n{bgr.shape[1]}x{bgr.shape[0]}", fontproperties=font, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return inputs


def show_orientation_predictions(
    model,
    image_path: str | Path,
    class_names: list[str],
    config_path: str | Path = "config.toml",
    conf: float = 0.45,
    include_square: bool = True,
    out_dir: str | Path = "notebook_outputs/orientation_crop_inputs",
) -> list[dict[str, Any]]:
    """Create landscape/portrait center-crops, run prediction and show inline."""
    font = configure_matplotlib_cyrillic()
    paths = save_orientation_inputs(
        image_path,
        out_dir=out_dir,
        config_path=config_path,
        include_square=include_square,
    )
    cols = len(paths)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    axes = np.array(axes).reshape(-1)
    reports: list[dict[str, Any]] = []
    for ax, (name, path) in zip(axes, paths):
        drawn_bgr, report = predict_image_with_meta(model, path, class_names, conf=conf)
        drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(drawn_rgb)
        ax.set_title(
            f"{name}\ninput={report['original_size'][0]}x{report['original_size'][1]}, detections={len(report['detections'])}",
            fontproperties=font,
            fontsize=10,
        )
        ax.axis("off")
        reports.append(report)
    plt.tight_layout()
    plt.show()
    return reports
