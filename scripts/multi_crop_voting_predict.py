#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_chess.dataset import load_dataset_info
from yolo_chess.infer import build_inference_model
from yolo_chess.multi_crop_voting import save_multicrop_voting_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-crop voting prediction for YOLOv3 chess detector")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--data", default="chess_yolo/data.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="runs/detect/multicrop_voting/prediction.jpg")
    parser.add_argument("--json", default="runs/detect/multicrop_voting/prediction.json")
    args = parser.parse_args()

    info = load_dataset_info(args.data, config_path=args.config)
    model = build_inference_model(args.weights, info.num_classes)

    report = save_multicrop_voting_prediction(
        model=model,
        image_path=args.image,
        out_image_path=args.out,
        out_json_path=args.json,
        class_names=info.class_names,
        config_path=args.config,
    )

    print("Saved image:", args.out)
    print("Saved json:", args.json)
    print("Detections:", len(report["detections"]))
    for det in report["detections"]:
        print(
            det["class_id"],
            det["class_name"],
            f"final={det['final_score']:.3f}",
            f"mean={det['mean_score']:.3f}",
            f"votes={det['votes']}/{det['support_denominator']}",
            det["box_restored_original_xyxy"],
        )


if __name__ == "__main__":
    main()
