from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from .font_utils import get_pil_cyrillic_font
from .infer import _predict_bgr_with_meta, _iou_xyxy

try:
    import tomllib  # py>=3.11
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


SupportDenominator = Literal["eligible", "total"]


@dataclass(frozen=True)
class CropSpec:
    """Square crop window in original-image pixel coordinates."""

    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    fraction: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def contains_point(self, x: float, y: float, pad: float = 1.0) -> bool:
        return (self.x1 - pad) <= x <= (self.x2 + pad) and (self.y1 - pad) <= y <= (self.y2 + pad)

    def as_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class MultiCropVotingConfig:
    """Parameters for multi-crop voting / weighted box fusion."""

    # Crops are square windows from the original square/rectangular input.
    # 1.0 = full min(width, height) window; smaller values zoom into local regions.
    crop_fractions: tuple[float, ...] = (1.0, 0.92, 0.84, 0.76)

    # center_and_cardinal: center + left/right/up/down for each fraction.
    # grid3: 3x3 crop grid for each fraction.
    # center: only centered crops for each fraction.
    offset_mode: str = "center_and_cardinal"

    # Low base confidence is intentional: weak detections can be confirmed by voting.
    base_conf: float = 0.12

    # IoU threshold for clustering detections of the same class.
    fusion_iou: float = 0.35

    # Keep only clusters with enough unique crop votes.
    min_votes: int = 1

    # Keep only clusters with enough final score.
    min_final_score: float = 0.18

    # How to normalize votes.
    # eligible: votes / crops whose window covers the fused box center.
    # total: votes / total crop count.
    support_denominator: SupportDenominator = "eligible"

    # final_score = score_weight * mean_score + support_weight * support_ratio
    score_weight: float = 0.55
    support_weight: float = 0.45

    # Transparency range for drawing final boxes.
    alpha_min: float = 0.18
    alpha_max: float = 0.85

    # Draw individual crop windows on debug images.
    draw_crop_windows: bool = False


def _read_toml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_multicrop_voting_config(config_path: str | Path = "config.toml") -> MultiCropVotingConfig:
    """Load [multi_crop_voting] from config.toml, keeping defaults for missing keys."""
    data = _read_toml(config_path).get("multi_crop_voting", {})
    cfg = MultiCropVotingConfig()

    crop_fractions = data.get("crop_fractions", cfg.crop_fractions)
    if isinstance(crop_fractions, list):
        crop_fractions = tuple(float(x) for x in crop_fractions)

    support_denominator = str(data.get("support_denominator", cfg.support_denominator))
    if support_denominator not in {"eligible", "total"}:
        support_denominator = cfg.support_denominator

    return MultiCropVotingConfig(
        crop_fractions=tuple(float(x) for x in crop_fractions),
        offset_mode=str(data.get("offset_mode", cfg.offset_mode)),
        base_conf=float(data.get("base_conf", cfg.base_conf)),
        fusion_iou=float(data.get("fusion_iou", cfg.fusion_iou)),
        min_votes=int(data.get("min_votes", cfg.min_votes)),
        min_final_score=float(data.get("min_final_score", cfg.min_final_score)),
        support_denominator=support_denominator,  # type: ignore[arg-type]
        score_weight=float(data.get("score_weight", cfg.score_weight)),
        support_weight=float(data.get("support_weight", cfg.support_weight)),
        alpha_min=float(data.get("alpha_min", cfg.alpha_min)),
        alpha_max=float(data.get("alpha_max", cfg.alpha_max)),
        draw_crop_windows=bool(data.get("draw_crop_windows", cfg.draw_crop_windows)),
    )


def _clamp_window(x1: float, y1: float, side: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a square window to image boundaries."""
    side = int(max(8, min(side, width, height)))
    x1 = int(round(x1))
    y1 = int(round(y1))

    x1 = max(0, min(width - side, x1))
    y1 = max(0, min(height - side, y1))
    return x1, y1, x1 + side, y1 + side


def _unique_windows(windows: Iterable[CropSpec]) -> list[CropSpec]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[CropSpec] = []
    for w in windows:
        key = (w.x1, w.y1, w.x2, w.y2)
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def make_square_multicrop_windows(
    width: int,
    height: int,
    crop_fractions: Iterable[float] = (1.0, 0.92, 0.84, 0.76),
    offset_mode: str = "center_and_cardinal",
) -> list[CropSpec]:
    """Create square crop windows for multi-crop voting.

    For a square 416x416 image this gives overlapping zoomed views such as:
    full frame, center crop, left/right/up/down crops, or 3x3 grid crops.

    For rectangular images the largest side is min(width, height), so inference
    still sees square crops that preserve object scale.
    """
    if width <= 0 or height <= 0:
        return []

    base_side = min(width, height)
    windows: list[CropSpec] = []

    for frac in crop_fractions:
        frac = float(frac)
        if frac <= 0:
            continue
        side = int(round(base_side * min(1.0, frac)))
        side = max(8, min(side, width, height))

        cx = (width - side) / 2.0
        cy = (height - side) / 2.0
        max_dx = max(0.0, (width - side) / 2.0)
        max_dy = max(0.0, (height - side) / 2.0)

        mode = offset_mode.lower().strip()

        if mode == "center":
            offsets = [(0.0, 0.0, "center")]

        elif mode == "grid3":
            offsets = []
            for oy, y_name in [(-max_dy, "top"), (0.0, "center"), (max_dy, "bottom")]:
                for ox, x_name in [(-max_dx, "left"), (0.0, "center"), (max_dx, "right")]:
                    offsets.append((ox, oy, f"{y_name}_{x_name}"))

        else:
            # Good default: not too many crops, but enough repeated evidence.
            offsets = [
                (0.0, 0.0, "center"),
                (-max_dx, 0.0, "left"),
                (max_dx, 0.0, "right"),
                (0.0, -max_dy, "top"),
                (0.0, max_dy, "bottom"),
            ]

        for ox, oy, suffix in offsets:
            x1, y1, x2, y2 = _clamp_window(cx + ox, cy + oy, side, width, height)
            windows.append(CropSpec(name=f"f{frac:.2f}_{suffix}", x1=x1, y1=y1, x2=x2, y2=y2, fraction=frac))

    # Put full-frame first if present. This makes debugging easier.
    windows = _unique_windows(windows)
    windows.sort(key=lambda w: (-(w.width * w.height), w.y1, w.x1, w.name))
    return windows


def _box_center(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _eligible_crop_count(box: Iterable[float], windows: list[CropSpec]) -> int:
    cx, cy = _box_center(box)
    return max(1, sum(1 for w in windows if w.contains_point(cx, cy)))


def _weighted_average_box(dets: list[dict[str, Any]]) -> list[float]:
    boxes = np.asarray([d["box_restored_original_xyxy"] for d in dets], dtype=np.float32)
    weights = np.asarray([max(1e-6, float(d.get("score", 0.0))) for d in dets], dtype=np.float32)
    fused = (boxes * weights[:, None]).sum(axis=0) / weights.sum()
    x1, y1, x2, y2 = fused.tolist()
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [float(x1), float(y1), float(x2), float(y2)]


def fuse_detections_by_voting(
    detections: list[dict[str, Any]],
    windows: list[CropSpec],
    class_names: list[str],
    fusion_iou: float = 0.35,
    min_votes: int = 1,
    min_final_score: float = 0.18,
    support_denominator: SupportDenominator = "eligible",
    score_weight: float = 0.55,
    support_weight: float = 0.45,
) -> list[dict[str, Any]]:
    """Class-aware IoU clustering + weighted box fusion + voting confidence."""
    if not detections:
        return []

    fused: list[dict[str, Any]] = []
    total_crops = max(1, len(windows))

    for class_id in sorted({int(d["class_id"]) for d in detections}):
        remaining = [d for d in detections if int(d["class_id"]) == class_id]
        remaining.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

        while remaining:
            seed = remaining.pop(0)
            seed_box = np.asarray(seed["box_restored_original_xyxy"], dtype=np.float32)

            cluster = [seed]
            survivors: list[dict[str, Any]] = []
            for d in remaining:
                box = np.asarray(d["box_restored_original_xyxy"], dtype=np.float32)
                if _iou_xyxy(seed_box, box) >= fusion_iou:
                    cluster.append(d)
                else:
                    survivors.append(d)
            remaining = survivors

            source_indices = sorted({int(d.get("crop_index", -1)) for d in cluster})
            votes = len(source_indices)
            if votes < min_votes:
                continue

            fused_box = _weighted_average_box(cluster)
            scores = [float(d.get("score", 0.0)) for d in cluster]
            mean_score = float(np.mean(scores)) if scores else 0.0
            max_score = float(np.max(scores)) if scores else 0.0

            eligible = _eligible_crop_count(fused_box, windows)
            if support_denominator == "total":
                support_ratio = votes / total_crops
                support_den = total_crops
            else:
                support_ratio = votes / max(1, eligible)
                support_den = eligible
            support_ratio = float(np.clip(support_ratio, 0.0, 1.0))

            total_support_ratio = float(np.clip(votes / total_crops, 0.0, 1.0))
            final_score = float(np.clip(score_weight * mean_score + support_weight * support_ratio, 0.0, 1.0))

            if final_score < min_final_score:
                continue

            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
            fused.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "score": final_score,
                    "final_score": final_score,
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "votes": votes,
                    "support_denominator": support_den,
                    "support_ratio": support_ratio,
                    "total_crops": total_crops,
                    "total_support_ratio": total_support_ratio,
                    "box_restored_original_xyxy": fused_box,
                    "source_crop_indices": source_indices,
                    "source_crop_names": [windows[i].name for i in source_indices if 0 <= i < len(windows)],
                    "cluster_size": len(cluster),
                    "cluster_scores": scores,
                    "inference_mode": "multi_crop_voting",
                }
            )

    fused.sort(key=lambda d: float(d["final_score"]), reverse=True)
    return fused


def _class_color_rgb(class_id: int) -> tuple[int, int, int]:
    """Deterministic bright-ish RGB color per class."""
    # Simple integer hash -> stable hue-ish RGB without external dependencies.
    rng = np.random.default_rng(class_id * 9973 + 17)
    color = rng.integers(80, 256, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def draw_voted_detections_bgr(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    alpha_min: float = 0.18,
    alpha_max: float = 0.85,
    draw_filled: bool = True,
    draw_outline: bool = True,
    font_size: int = 16,
    draw_votes: bool = True,
) -> np.ndarray:
    """Draw fused boxes with transparent fill depending on voting confidence."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_pil_cyrillic_font(size=font_size)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_restored_original_xyxy"]]
        if x2 <= x1 or y2 <= y1:
            continue

        final_score = float(det.get("final_score", det.get("score", 0.0)))
        alpha = alpha_min + (alpha_max - alpha_min) * float(np.clip(final_score, 0.0, 1.0))
        alpha_u8 = int(round(255 * float(np.clip(alpha, 0.0, 1.0))))

        color = _class_color_rgb(int(det.get("class_id", 0)))
        fill = (*color, max(25, int(alpha_u8 * 0.35)))
        outline = (*color, alpha_u8)

        if draw_filled:
            draw.rectangle([x1, y1, x2, y2], fill=fill)
        if draw_outline:
            for k in range(3):
                draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=outline)

        if draw_votes:
            text = (
                f'{det["class_name"]} {final_score:.2f} | '
                f'{int(det.get("votes", 0))}/{int(det.get("support_denominator", det.get("total_crops", 1)))}'
            )
        else:
            text = f'{det["class_name"]} {final_score:.2f}'

        tb = draw.textbbox((x1, y1), text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=(*color, max(180, alpha_u8)))
        draw.text((x1 + 4, ty + 3), text, fill=(255, 255, 255, 255), font=font)

    composed = Image.alpha_composite(pil, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def draw_crop_windows_bgr(
    image_bgr: np.ndarray,
    windows: list[CropSpec],
    font_size: int = 12,
) -> np.ndarray:
    """Debug drawing of all crop windows."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_pil_cyrillic_font(size=font_size)

    for idx, w in enumerate(windows):
        color = _class_color_rgb(idx)
        outline = (*color, 180)
        for k in range(2):
            draw.rectangle([w.x1 - k, w.y1 - k, w.x2 + k, w.y2 + k], outline=outline)
        draw.text((w.x1 + 3, w.y1 + 3), f"{idx}:{w.name}", fill=outline, font=font)

    composed = Image.alpha_composite(pil, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def predict_multicrop_voting_bgr(
    model: tf.keras.Model,
    image_bgr: np.ndarray,
    class_names: list[str],
    config: MultiCropVotingConfig | None = None,
    image_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run multi-crop detection, fuse repeated boxes, and draw transparent boxes.

    Returns:
      drawn_bgr, report
    """
    if config is None:
        config = MultiCropVotingConfig()

    height, width = image_bgr.shape[:2]
    windows = make_square_multicrop_windows(
        width=width,
        height=height,
        crop_fractions=config.crop_fractions,
        offset_mode=config.offset_mode,
    )

    all_detections: list[dict[str, Any]] = []
    crop_reports: list[dict[str, Any]] = []

    for idx, w in enumerate(windows):
        crop = image_bgr[w.y1 : w.y2, w.x1 : w.x2]
        _, crop_report = _predict_bgr_with_meta(
            model,
            crop,
            class_names,
            conf=config.base_conf,
            image_path=f"{image_path or '<array>'}#crop_{idx}_{w.name}_{w.x1}_{w.y1}_{w.x2}_{w.y2}",
        )

        mapped: list[dict[str, Any]] = []
        for det in crop_report["detections"]:
            bx1, by1, bx2, by2 = [float(v) for v in det["box_restored_original_xyxy"]]
            full_box = [
                float(np.clip(bx1 + w.x1, 0, width - 1)),
                float(np.clip(by1 + w.y1, 0, height - 1)),
                float(np.clip(bx2 + w.x1, 0, width - 1)),
                float(np.clip(by2 + w.y1, 0, height - 1)),
            ]
            if full_box[2] <= full_box[0] or full_box[3] <= full_box[1]:
                continue
            det2 = dict(det)
            det2["box_restored_original_xyxy"] = full_box
            det2["crop_index"] = idx
            det2["crop_name"] = w.name
            det2["source_crop_xyxy"] = w.as_xyxy()
            det2["raw_crop_score"] = float(det.get("score", 0.0))
            mapped.append(det2)
            all_detections.append(det2)

        crop_reports.append(
            {
                "crop_index": idx,
                "crop_name": w.name,
                "crop_xyxy": w.as_xyxy(),
                "fraction": w.fraction,
                "detections": mapped,
                "detections_count": len(mapped),
            }
        )

    fused = fuse_detections_by_voting(
        all_detections,
        windows=windows,
        class_names=class_names,
        fusion_iou=config.fusion_iou,
        min_votes=config.min_votes,
        min_final_score=config.min_final_score,
        support_denominator=config.support_denominator,
        score_weight=config.score_weight,
        support_weight=config.support_weight,
    )

    drawn = draw_voted_detections_bgr(
        image_bgr.copy(),
        fused,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
    )

    if config.draw_crop_windows:
        drawn = draw_crop_windows_bgr(drawn, windows)

    report = {
        "image_path": str(image_path) if image_path is not None else None,
        "original_size": [width, height],
        "network_size": [416, 416],
        "inference_mode": "multi_crop_voting",
        "config": asdict(config),
        "windows": [asdict(w) | {"xyxy": w.as_xyxy()} for w in windows],
        "crop_reports": crop_reports,
        "detections_before_fusion_count": len(all_detections),
        "detections": fused,
    }
    return drawn, report


def predict_multicrop_voting_image(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    config_path: str | Path = "config.toml",
    config: MultiCropVotingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Path-based wrapper around predict_multicrop_voting_bgr."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if config is None:
        config = load_multicrop_voting_config(config_path)
    return predict_multicrop_voting_bgr(model, image_bgr, class_names, config=config, image_path=image_path)


def show_multicrop_voting_inline(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    config_path: str | Path = "config.toml",
    figsize: tuple[int, int] = (9, 9),
) -> dict[str, Any]:
    """Notebook helper: run multi-crop voting and show result with matplotlib."""
    import matplotlib.pyplot as plt

    drawn_bgr, report = predict_multicrop_voting_image(
        model,
        image_path,
        class_names,
        config_path=config_path,
    )
    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(drawn_rgb)
    plt.axis("off")
    plt.title(
        f"multi-crop voting | detections={len(report['detections'])} | "
        f"raw={report['detections_before_fusion_count']} | crops={len(report['windows'])}"
    )
    plt.show()
    return report


def show_multicrop_debug_grid_inline(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    config_path: str | Path = "config.toml",
    max_crops: int = 16,
    cols: int = 4,
) -> dict[str, Any]:
    """Notebook helper: show individual crop detections before fusion."""
    import math
    import matplotlib.pyplot as plt

    cfg = load_multicrop_voting_config(config_path)
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image_bgr.shape[:2]
    windows = make_square_multicrop_windows(w, h, cfg.crop_fractions, cfg.offset_mode)
    shown = windows[:max_crops]
    rows = math.ceil(len(shown) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, crop_spec in zip(axes, shown):
        crop = image_bgr[crop_spec.y1 : crop_spec.y2, crop_spec.x1 : crop_spec.x2]
        drawn_bgr, crop_report = _predict_bgr_with_meta(
            model,
            crop,
            class_names,
            conf=cfg.base_conf,
            image_path=f"{image_path}#{crop_spec.name}",
        )
        drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(drawn_rgb)
        ax.set_title(f"{crop_spec.name}\ndets={len(crop_report['detections'])}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(shown) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    _, report = predict_multicrop_voting_image(model, image_path, class_names, config_path=config_path, config=cfg)
    return report


def save_multicrop_voting_prediction(
    model: tf.keras.Model,
    image_path: str | Path,
    out_image_path: str | Path,
    out_json_path: str | Path | None,
    class_names: list[str],
    config_path: str | Path = "config.toml",
) -> dict[str, Any]:
    drawn_bgr, report = predict_multicrop_voting_image(model, image_path, class_names, config_path=config_path)
    out_image_path = Path(out_image_path)
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image_path), drawn_bgr)

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
