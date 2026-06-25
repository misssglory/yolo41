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

    Returns a 416x416 image with padding and metadata needed to restore boxes.
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

    Returns: (box_net_416_xyxy, box_original_xyxy)
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
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/run/current-system/sw/share/X11/fonts/TTF/DejaVuSans.ttf",
        "/run/current-system/sw/share/fonts/truetype/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


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


def predict_image_with_meta(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: list[str],
    conf: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict on any image size and draw boxes restored to original image size."""
    image_path = Path(image_path)
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    letterboxed_bgr, meta = letterbox_image_bgr(original_bgr, SIZE)
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

    drawn = _draw_unicode_detections_bgr(original_bgr.copy(), detections)

    report = {
        "image_path": str(image_path),
        "original_size": [meta.original_width, meta.original_height],
        "network_size": [meta.network_width, meta.network_height],
        "output_size": [int(drawn.shape[1]), int(drawn.shape[0])],
        "letterbox": asdict(meta),
        "detections": detections,
    }
    return drawn, report


def save_prediction(
    model: tf.keras.Model,
    image_path: str | Path,
    out_image_path: str | Path,
    out_json_path: str | Path | None,
    class_names: list[str],
    conf: float = 0.25,
) -> dict[str, Any]:
    out_image_path = Path(out_image_path)
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    image, report = predict_image_with_meta(model, image_path, class_names, conf)
    cv2.imwrite(str(out_image_path), image)

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
