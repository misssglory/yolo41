#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_chess.gt_viz import collect_images_from_split, draw_ground_truth_matplotlib


def parse_args():
    p = argparse.ArgumentParser(description="Draw ground-truth chess labels with matplotlib.")
    p.add_argument("--data", default="chess_yolo/data.yaml")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"])
    p.add_argument("--out", default="runs/detect/ground_truth")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--sample-count", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    images, class_names, _ = collect_images_from_split(args.data, args.split, args.config)
    if not images:
        raise RuntimeError(f"No images found for split={args.split!r}")
    random.seed(args.seed)
    samples = random.sample(images, min(args.sample_count, len(images)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(samples, start=1):
        fig, ax, objects, label_path = draw_ground_truth_matplotlib(
            image_path,
            class_names,
            config_path=args.config,
            draw_imgsz=args.imgsz,
            show_title=True,
            fontsize=8,
        )
        out_path = out / f"{idx:02d}_{Path(image_path).stem}_gt.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"{image_path.name}: objects={len(objects)} label={label_path} -> {out_path}")


if __name__ == "__main__":
    main()
