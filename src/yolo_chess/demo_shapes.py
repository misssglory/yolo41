from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_SUFFIXES, load_config, orientation_cases
from .dataset import load_dataset_info
from .download import ensure_data_yaml
from .infer import build_inference_model, save_prediction


def first_image(source: Path) -> Path:
    if source.is_file():
        return source
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return p
    raise RuntimeError(f"No image found in {source}")


def resize_cover_center_crop(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize without distortion: cover target rectangle and crop center.

    This is intentionally different from cv2.resize(img, (w, h)), which stretches pieces and
    can create false positives. Here chess pieces keep their proportions.
    """
    src_h, src_w = img.shape[:2]
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    cropped = resized[top : top + target_h, left : left + target_w]

    # Defensive fallback for odd rounding edge-cases.
    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def resize_contain_letterbox(img: np.ndarray, target_w: int, target_h: int, pad_value: int = 114) -> np.ndarray:
    """Resize without distortion: contain full image and pad to target rectangle."""
    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=img.dtype)
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def make_oriented_image(img: np.ndarray, target_w: int, target_h: int, transform: str) -> np.ndarray:
    transform = transform.strip().lower()
    if transform in {"center_crop", "crop", "cover", "cover_crop"}:
        return resize_cover_center_crop(img, target_w, target_h)
    if transform in {"letterbox", "pad", "contain"}:
        return resize_contain_letterbox(img, target_w, target_h)
    raise ValueError(
        f"Unsupported [demo].rectangular_transform={transform!r}. "
        "Use 'center_crop' or 'letterbox'."
    )


def make_shape_inputs(base_image: Path, out_dir: Path, cases: list[tuple[str, tuple[int, int]]], transform: str) -> list[tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(base_image))
    if img is None:
        raise FileNotFoundError(base_image)

    result: list[tuple[str, Path]] = []
    for name, (w, h) in cases:
        # Even the square baseline is generated through the same non-distorting path.
        oriented = make_oriented_image(img, int(w), int(h), transform)
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), oriented)
        result.append((name, path))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate restoring YOLO outputs to original image sizes.")
    parser.add_argument("--weights", required=True, help="Path to *.weights.h5")
    parser.add_argument("--data", default="chess_yolo/data.yaml")
    parser.add_argument("--source", default="chess_yolo/test", help="Image or directory with test images")
    parser.add_argument("--out", default="runs/detect/demo_shapes")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument(
        "--orientation-mode",
        default=None,
        help="Override [demo].orientation_mode: square | mixed",
    )
    parser.add_argument(
        "--rectangular-transform",
        default=None,
        help="Override [demo].rectangular_transform: center_crop | letterbox",
    )
    parser.add_argument("--download-if-missing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides are useful for quick experiments while keeping config.toml as source of truth.
    demo_cfg = cfg.demo
    if args.orientation_mode is not None or args.rectangular_transform is not None:
        from .config import DemoConfig

        demo_cfg = DemoConfig(
            orientation_mode=(args.orientation_mode or demo_cfg.orientation_mode).strip().lower(),
            rectangular_transform=(args.rectangular_transform or demo_cfg.rectangular_transform).strip().lower(),
            square_size=demo_cfg.square_size,
            landscape_size=demo_cfg.landscape_size,
            portrait_size=demo_cfg.portrait_size,
        )

    cases = orientation_cases(demo_cfg)

    data_yaml = ensure_data_yaml(args.data, download_if_missing=args.download_if_missing)
    info = load_dataset_info(data_yaml, config_path=args.config)
    model = build_inference_model(args.weights, info.num_classes)

    out = Path(args.out)
    input_dir = out / "inputs"
    metadata_dir = out / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    base = first_image(Path(args.source))
    shape_inputs = make_shape_inputs(base, input_dir, cases, demo_cfg.rectangular_transform)

    rows = []
    for name, image_path in shape_inputs:
        out_image = out / f"{name}_pred.jpg"
        out_json = metadata_dir / f"{name}.json"
        report = save_prediction(model, image_path, out_image, out_json, info.class_names, args.conf)
        letterbox = report["letterbox"]
        rows.append(
            {
                "case": name,
                "orientation_mode": demo_cfg.orientation_mode,
                "rectangular_transform": demo_cfg.rectangular_transform,
                "input_image": str(image_path),
                "output_image": str(out_image),
                "original_width": report["original_size"][0],
                "original_height": report["original_size"][1],
                "network_width": report["network_size"][0],
                "network_height": report["network_size"][1],
                "output_width": report["output_size"][0],
                "output_height": report["output_size"][1],
                "scale": letterbox["scale"],
                "pad_x": letterbox["pad_x"],
                "pad_y": letterbox["pad_y"],
                "detections": len(report["detections"]),
                "first_box_net_416_xyxy": json.dumps(report["detections"][0]["box_net_416_xyxy"]) if report["detections"] else "",
                "first_box_restored_original_xyxy": json.dumps(report["detections"][0]["box_restored_original_xyxy"]) if report["detections"] else "",
            }
        )
        print(
            f"{name}: original={report['original_size']}, network={report['network_size']}, "
            f"output={report['output_size']}, transform={demo_cfg.rectangular_transform}, "
            f"scale={letterbox['scale']:.6f}, pad=({letterbox['pad_x']},{letterbox['pad_y']}), "
            f"detections={len(report['detections'])}"
        )

    report_csv = out / "restoration_report.csv"
    if rows:
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    effective_config = {
        "config_path": str(args.config),
        "image_size": cfg.image_size,
        "demo": asdict(demo_cfg),
        "cases": [{"name": name, "size": [size[0], size[1]]} for name, size in cases],
    }
    (out / "effective_config.json").write_text(json.dumps(effective_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved restoration report: {report_csv}")
    print(f"Saved effective config:    {out / 'effective_config.json'}")


if __name__ == "__main__":
    main()
