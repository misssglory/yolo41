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

# Class order used by the lesson chess_yolo dataset.
# It must match numeric class ids in YOLO label .txt files.
DEFAULT_CHESS_CLASS_NAMES = [
    "слон",
    "черный слон",
    "черный король",
    "черный конь",
    "черная пешка",
    "черный ферзь",
    "черная ладья",
    "белый слон",
    "белый король",
    "белый конь",
    "белая пешка",
    "белый ферзь",
    "белая ладья",
]


@dataclass(frozen=True)
class DemoConfig:
    """Controls the orientation demonstration required by the homework."""

    orientation_mode: str = "square"
    rectangular_transform: str = "center_crop"
    square_size: tuple[int, int] = (416, 416)
    landscape_size: tuple[int, int] = (800, 500)
    portrait_size: tuple[int, int] = (500, 900)
    sample_count: int = 16
    grid_cols: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    """Controls initialization before fine-tuning."""

    weights: str = "auto"
    use_darknet_pretrained: bool = True
    darknet_weights: str = "yolov3.weights"
    darknet_weights_url: str = "https://storage.yandexcloud.net/academy.ai/CV/yolov3.weights"


@dataclass(frozen=True)
class EnvironmentConfig:
    """Controls dependency installation in notebooks/Colab helper cells.

    The Nix flake can still use uv independently. This config is meant for
    notebook cells and scripts/install_deps.py, where Colab often works better
    with plain pip even if an uv binary happens to be present.
    """

    use_uv: bool = True
    requirements: str = "requirements.txt"
    editable_install: bool = True
    upgrade_pip: bool = False


@dataclass(frozen=True)
class LabelsConfig:
    """Controls class-name mapping for labels and annotations.

    class_names_override is optional. If empty, names are read from data.yaml.
    It is useful in notebooks when data.yaml was damaged or saved with wrong encoding.
    """

    class_names_override: tuple[str, ...] = ()
    use_default_chess_names_if_nc_13: bool = True


@dataclass(frozen=True)
class AppConfig:
    image_size: int = SIZE
    demo: DemoConfig = field(default_factory=DemoConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)


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
    training = raw.get("training", {}) or {}
    environment = raw.get("environment", {}) or {}
    labels = raw.get("labels", {}) or {}

    image_size = int(image.get("size", SIZE))
    demo_cfg = DemoConfig(
        orientation_mode=str(demo.get("orientation_mode", "square")).strip().lower(),
        rectangular_transform=str(demo.get("rectangular_transform", "center_crop")).strip().lower(),
        square_size=_pair(demo.get("square_size"), (image_size, image_size)),
        landscape_size=_pair(demo.get("landscape_size"), (800, 500)),
        portrait_size=_pair(demo.get("portrait_size"), (500, 900)),
        sample_count=int(demo.get("sample_count", 16)),
        grid_cols=int(demo.get("grid_cols", 4)),
    )
    training_cfg = TrainingConfig(
        weights=str(training.get("weights", "auto")).strip(),
        use_darknet_pretrained=bool(training.get("use_darknet_pretrained", True)),
        darknet_weights=str(training.get("darknet_weights", "yolov3.weights")).strip(),
        darknet_weights_url=str(
            training.get(
                "darknet_weights_url",
                "https://storage.yandexcloud.net/academy.ai/CV/yolov3.weights",
            )
        ).strip(),
    )
    env_cfg = EnvironmentConfig(
        use_uv=bool(environment.get("use_uv", True)),
        requirements=str(environment.get("requirements", "requirements.txt")).strip(),
        editable_install=bool(environment.get("editable_install", True)),
        upgrade_pip=bool(environment.get("upgrade_pip", False)),
    )
    raw_override = labels.get("class_names_override", []) or []
    labels_cfg = LabelsConfig(
        class_names_override=tuple(str(x) for x in raw_override),
        use_default_chess_names_if_nc_13=bool(labels.get("use_default_chess_names_if_nc_13", True)),
    )
    return AppConfig(
        image_size=image_size,
        demo=demo_cfg,
        training=training_cfg,
        environment=env_cfg,
        labels=labels_cfg,
    )


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
