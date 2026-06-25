from __future__ import annotations

import argparse
from pathlib import Path

from .config import IMAGE_SUFFIXES
from .dataset import load_dataset_info
from .download import ensure_data_yaml
from .infer import build_inference_model, save_prediction


def collect_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    images = []
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            images.append(p)
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv3 chess inference with original-size output restoration.")
    parser.add_argument("--weights", required=True, help="Path to *.weights.h5")
    parser.add_argument("--data", default="chess_yolo/data.yaml", help="data.yaml with class names")
    parser.add_argument("--source", default="chess_yolo/test", help="Image file or directory")
    parser.add_argument("--out", default="runs/detect/predict")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--download-if-missing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = ensure_data_yaml(args.data, download_if_missing=args.download_if_missing)
    info = load_dataset_info(data_yaml)
    model = build_inference_model(args.weights, info.num_classes)

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = collect_sources(source)
    if not images:
        raise RuntimeError(f"No images found in source: {source}")

    for image_path in images:
        out_image = out / f"{image_path.stem}_pred.jpg"
        out_json = out / "metadata" / f"{image_path.stem}.json"
        report = save_prediction(model, image_path, out_image, out_json, info.class_names, args.conf)
        print(
            f"{image_path.name}: original={report['original_size']}, "
            f"network={report['network_size']}, output={report['output_size']}, "
            f"detections={len(report['detections'])} -> {out_image}"
        )


if __name__ == "__main__":
    main()
