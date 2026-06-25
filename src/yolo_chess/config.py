from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

import numpy as np

SIZE = 416
ANCHORS = np.array(
    [
        (10, 13),
        (16, 30),
        (33, 23),
        (30, 61),
        (62, 45),
        (59, 119),
        (116, 90),
        (156, 198),
        (373, 326),
    ],
    np.float32,
) / SIZE

ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]], dtype=np.int32)

YOLO_IOU_THRESHOLD = 0.5
YOLO_SCORE_THRESHOLD = 0.25
MAX_BOXES = 100

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DemoConfig:
    """Controls the orientation demonstration required by the homework."""

    orientation_mode: str = "square"
    rectangular_transform: str = "center_crop"
    square_size: tuple[int, int] = (416, 416)
    landscape_size: tuple[int, int] = (800, 500)
    portrait_size: tuple[int, int] = (500, 900)


@dataclass(frozen=True)
class AppConfig:
    image_size: int = SIZE
    demo: DemoConfig = field(default_factory=DemoConfig)


def _pair(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected [width, height], got: {value!r}")


def load_config(path: str | Path = "config.toml") -> AppConfig:
    """Load config.toml if it exists; otherwise return safe defaults.

    The assignment dataset is 416x416, so the default mode is square-only baseline.
    Switch [demo].orientation_mode to "mixed" to also create landscape/portrait inputs.
    """
    path = Path(path)
    if not path.exists():
        return AppConfig()

    with path.open("rb") as f:
        raw = tomllib.load(f)

    image = raw.get("image", {}) or {}
    demo = raw.get("demo", {}) or {}

    image_size = int(image.get("size", SIZE))
    demo_cfg = DemoConfig(
        orientation_mode=str(demo.get("orientation_mode", "square")).strip().lower(),
        rectangular_transform=str(demo.get("rectangular_transform", "center_crop")).strip().lower(),
        square_size=_pair(demo.get("square_size"), (image_size, image_size)),
        landscape_size=_pair(demo.get("landscape_size"), (800, 500)),
        portrait_size=_pair(demo.get("portrait_size"), (500, 900)),
    )
    return AppConfig(image_size=image_size, demo=demo_cfg)


def orientation_cases(demo: DemoConfig) -> list[tuple[str, tuple[int, int]]]:
    """Return demo cases as (case_name, (width, height))."""
    mode = demo.orientation_mode.strip().lower()
    if mode in {"square", "baseline", "baseline_square", "only_square"}:
        return [("original_416_square", demo.square_size)]
    if mode in {"mixed", "all", "portrait_landscape", "landscape_portrait", "rectangular"}:
        return [
            ("original_416_square", demo.square_size),
            ("landscape_800x500", demo.landscape_size),
            ("portrait_500x900", demo.portrait_size),
        ]
    raise ValueError(
        f"Unsupported [demo].orientation_mode={demo.orientation_mode!r}. "
        "Use 'square' for baseline or 'mixed' for square + landscape + portrait."
    )
