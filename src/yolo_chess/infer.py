from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from .config import SIZE
from .font_utils import get_pil_cyrillic_font
from .model import YoloV3


@dataclass
class LetterboxMeta:
    original_width: int
    original_height: int
    network_width: int
    network_height: int
    scale: float
    pad_x: float
    pad_y: float
    resized_width: int
    resized_height: int


def letterbox_image_bgr(image_bgr: np.ndarray, size: int = SIZE) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize image to square network input while preserving aspect ratio.

    Returns a size x size image with padding and metadata needed to restore boxes.
    """
    original_height, original_width = image_bgr.shape[:2]
    scale = min(size / original_width, size / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    meta = LetterboxMeta(
        original_width=original_width,
        original_height=original_height,
        network_width=size,
        network_height=size,
        scale=float(scale),
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        resized_width=resized_width,
        resized_height=resized_height,
    )
    return canvas, meta


def restore_box_from_letterbox(box_norm: np.ndarray, meta: LetterboxMeta) -> tuple[list[float], list[float]]:
    """Convert normalized network xyxy box to original-image pixel xyxy.

    Returns: (box_net_xyxy, box_original_xyxy)
    """
    box = np.asarray(box_norm, dtype=np.float32)
    box_net = np.array(
        [
            box[0] * meta.network_width,
            box[1] * meta.network_height,
            box[2] * meta.network_width,
            box[3] * meta.network_height,
        ],
        dtype=np.float32,
    )

    restored = np.array(
        [
            (box_net[0] - meta.pad_x) / meta.scale,
            (box_net[1] - meta.pad_y) / meta.scale,
            (box_net[2] - meta.pad_x) / meta.scale,
            (box_net[3] - meta.pad_y) / meta.scale,
        ],
        dtype=np.float32,
    )
    restored[[0, 2]] = np.clip(restored[[0, 2]], 0, meta.original_width - 1)
    restored[[1, 3]] = np.clip(restored[[1, 3]], 0, meta.original_height - 1)
    return box_net.tolist(), restored.tolist()


def build_inference_model(weights: str | Path, num_classes: int) -> tf.keras.Model:
    """Load training weights into the inference model with NMS outputs."""
    weights = Path(weights)
    train_model = YoloV3(size=SIZE, classes=num_classes, training=True)
    train_model.load_weights(str(weights))

    inference_model = YoloV3(size=SIZE, classes=num_classes, training=False)
    inference_model.set_weights(train_model.get_weights())
    return inference_model


def _find_unicode_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find a font that can draw Cyrillic labels. OpenCV putText cannot."""
    return get_pil_cyrillic_font(size=size)


def _draw_unicode_detections_bgr(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    font_size: int = 16,
) -> np.ndarray:
    """Draw boxes and Cyrillic labels using PIL, then return BGR for cv2.imwrite."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = _find_unicode_font(font_size)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_restored_original_xyxy"]]
        if x2 <= x1 or y2 <= y1:
            continue
        text = f'{det["class_name"]} {float(det["score"]):.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        tb = draw.textbbox((x1, y1), text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=(255, 0, 0))
        draw.text((x1 + 4, ty + 3), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _predict_bgr_with_meta(
    model: tf.keras.Model,
    image_bgr: np.ndarray,
    class_names: list[str],
    conf: float = 0.25,
    image_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict on a BGR image array and draw boxes restored to that array's size."""
    letterboxed_bgr, meta = letterbox_image_bgr(image_bgr, SIZE)
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    batch = np.expand_dims(rgb, axis=0)

    batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
    # Direct call avoids Keras predict graph/XLA issues with CombinedNonMaxSuppression in Colab.
    with tf.device("/CPU:0"):
        boxes, scores, classes, valid = model(batch_tensor, training=False)

    boxes = boxes.numpy()[0]
    scores = scores.numpy()[0]
    classes = classes.numpy()[0].astype(np.int32)
    valid = int(valid.numpy()[0])

    detections: list[dict[str, Any]] = []

    for i in range(valid):
        score = float(scores[i])
        if score < conf:
            continue
        class_id = int(classes[i])
        box_net, box_original = restore_box_from_letterbox(boxes[i], meta)
        x1, y1, x2, y2 = [int(round(v)) for v in box_original]
        if x2 <= x1 or y2 <= y1:
            continue

        name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        detections.append(
            {
                "class_id": class_id,
                "class_name": name,
                "score": score,
                "box_net_416_xyxy": box_net,
                "box_restored_original_xyxy": box_original,
            }
        )

    drawn = _draw_unicode_detections_bgr(image_bgr.copy(), detections)

    report = {
        "image_path": str(image_path) if image_path is not None else None,
        "original_size": [meta.original_width, meta.original_height],
        "network_size": [meta.network_width, meta.network_height],
        "output_size": [int(drawn.shape[1]), int(drawn.shape[0])],
        "letterbox": asdict(meta),
        "detections": detections,
        "inference_mode": "letterbox_full_image",
    }
    return drawn, report


def predict_image_with_meta(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    conf: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict on any image size using full-image letterbox inference.

    This is faithful to a standard YOLO pipeline, but on very rectangular images
    the objects become smaller inside the 416x416 padded network input. For the
    homework orientation crop demo, use predict_image_with_crop_windows instead:
    it runs square crop windows and restores detections to the original rectangle.
    """
    image_path = Path(image_path)
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return _predict_bgr_with_meta(model, original_bgr, class_names, conf=conf, image_path=image_path)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _nms_detections_numpy(detections: list[dict[str, Any]], iou_threshold: float = 0.45) -> list[dict[str, Any]]:
    """Simple class-aware NMS for detections restored from multiple crop windows."""
    if not detections:
        return []
    kept: list[dict[str, Any]] = []
    for class_id in sorted({int(d["class_id"]) for d in detections}):
        cls_dets = [d for d in detections if int(d["class_id"]) == class_id]
        cls_dets.sort(key=lambda d: float(d["score"]), reverse=True)
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            best_box = np.asarray(best["box_restored_original_xyxy"], dtype=np.float32)
            survivors = []
            for d in cls_dets:
                box = np.asarray(d["box_restored_original_xyxy"], dtype=np.float32)
                if _iou_xyxy(best_box, box) < iou_threshold:
                    survivors.append(d)
            cls_dets = survivors
    kept.sort(key=lambda d: float(d["score"]), reverse=True)
    return kept


def _window_starts(length: int, window: int, overlap: float) -> list[int]:
    if length <= window:
        return [0]
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, max(1, length - window + 1), stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def make_square_crop_windows(
    width: int,
    height: int,
    crop_size: int | None = None,
    overlap: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    """Return square crop windows as (x1, y1, x2, y2) covering a rectangle.

    crop_size defaults to min(width, height). For landscape images this gives
    left/center/right square crops; for portrait images top/center/bottom crops.
    This keeps the object scale close to the original square training images and
    avoids the shrinkage caused by full-image letterbox inference.
    """
    if width <= 0 or height <= 0:
        return []
    if crop_size is None:
        crop_size = min(width, height)
    crop_size = int(max(8, min(crop_size, width, height)))
    xs = _window_starts(width, crop_size, overlap)
    ys = _window_starts(height, crop_size, overlap)
    return [(x, y, x + crop_size, y + crop_size) for y in ys for x in xs]


def predict_image_with_crop_windows(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    conf: float = 0.25,
    crop_size: int | None = None,
    overlap: float = 0.35,
    nms_iou: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict on landscape/portrait images with square crop windows.

    The model was trained on square 416x416 chess images. If a rectangular image
    is letterboxed as a whole, pieces may become too small and detections vanish.
    This function instead runs the model on overlapping square crops, then maps
    the detections back to the original rectangular image and merges duplicates.
    """
    image_path = Path(image_path)
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = original_bgr.shape[:2]
    windows = make_square_crop_windows(width, height, crop_size=crop_size, overlap=overlap)

    all_detections: list[dict[str, Any]] = []
    window_reports: list[dict[str, Any]] = []

    for idx, (x1, y1, x2, y2) in enumerate(windows):
        crop = original_bgr[y1:y2, x1:x2]
        _, crop_report = _predict_bgr_with_meta(
            model,
            crop,
            class_names,
            conf=conf,
            image_path=f"{image_path}#crop_{idx}_{x1}_{y1}_{x2}_{y2}",
        )
        mapped = []
        for det in crop_report["detections"]:
            bx1, by1, bx2, by2 = det["box_restored_original_xyxy"]
            full_box = [
                float(np.clip(bx1 + x1, 0, width - 1)),
                float(np.clip(by1 + y1, 0, height - 1)),
                float(np.clip(bx2 + x1, 0, width - 1)),
                float(np.clip(by2 + y1, 0, height - 1)),
            ]
            if full_box[2] <= full_box[0] or full_box[3] <= full_box[1]:
                continue
            det2 = dict(det)
            det2["box_restored_original_xyxy"] = full_box
            det2["source_window_xyxy"] = [x1, y1, x2, y2]
            all_detections.append(det2)
            mapped.append(det2)
        window_reports.append(
            {
                "window_index": idx,
                "window_xyxy": [x1, y1, x2, y2],
                "detections_before_merge": mapped,
            }
        )

    merged = _nms_detections_numpy(all_detections, iou_threshold=nms_iou)
    drawn = _draw_unicode_detections_bgr(original_bgr.copy(), merged)

    report = {
        "image_path": str(image_path),
        "original_size": [width, height],
        "output_size": [int(drawn.shape[1]), int(drawn.shape[0])],
        "network_size": [SIZE, SIZE],
        "inference_mode": "square_crop_windows",
        "crop_size": crop_size if crop_size is not None else min(width, height),
        "overlap": overlap,
        "nms_iou": nms_iou,
        "windows": [list(w) for w in windows],
        "window_reports": window_reports,
        "detections_before_merge_count": len(all_detections),
        "detections": merged,
    }
    return drawn, report


def predict_image_auto_orientation(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    conf: float = 0.25,
    crop_overlap: float = 0.35,
    nms_iou: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use full-image inference for square-ish inputs, crop-window inference otherwise."""
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = bgr.shape[:2]
    aspect = max(w / h, h / w)
    if aspect <= 1.15:
        return predict_image_with_meta(model, image_path, class_names, conf=conf)
    return predict_image_with_crop_windows(
        model,
        image_path,
        class_names,
        conf=conf,
        overlap=crop_overlap,
        nms_iou=nms_iou,
    )


def save_prediction(
    model: tf.keras.Model,
    image_path: str | Path,
    out_image_path: str | Path,
    out_json_path: str | Path | None,
    class_names: list[str],
    conf: float = 0.25,
    use_crop_windows: bool = False,
) -> dict[str, Any]:
    out_image_path = Path(out_image_path)
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    if use_crop_windows:
        image, report = predict_image_with_crop_windows(model, image_path, class_names, conf=conf)
    else:
        image, report = predict_image_with_meta(model, image_path, class_names, conf)
    cv2.imwrite(str(out_image_path), image)

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
