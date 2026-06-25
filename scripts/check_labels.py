#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_chess.config import DEFAULT_CHESS_CLASS_NAMES, IMAGE_SUFFIXES, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Check chess labels: class mapping + label row format.")
    p.add_argument("--data", default="chess_yolo/data.yaml")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--split", default="all", choices=["train", "val", "valid", "test", "all"])
    return p.parse_args()


def normalize_names(names, nc: int, config_path: str | Path) -> list[str]:
    cfg = load_config(config_path)
    if cfg.labels.class_names_override:
        out = list(cfg.labels.class_names_override)
    elif isinstance(names, dict):
        out = [f"class_{i}" for i in range(nc)]
        for key, value in names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < nc:
                out[idx] = str(value)
    elif isinstance(names, list):
        out = [str(x) for x in names]
    else:
        out = []

    if nc == 13 and cfg.labels.use_default_chess_names_if_nc_13:
        if len(out) != 13 or any(not str(x).strip() for x in out):
            out = list(DEFAULT_CHESS_CLASS_NAMES)
    if len(out) < nc:
        out.extend(f"class_{i}" for i in range(len(out), nc))
    return out[:nc]


def resolve_yaml_path(root: Path, value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(resolve_yaml_path(root, item))
        return result
    p = Path(str(value))
    if not p.is_absolute():
        p = root / p
    return [p]


def collect_images(paths: list[Path]) -> list[Path]:
    images = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            images.append(p)
        elif p.is_dir():
            images.extend(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(set(images))


def find_label_path(image_path: Path) -> Path:
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def row_to_bbox(parts: list[str], box_format: str):
    if len(parts) < 5:
        return None
    fmt = box_format.strip().lower()
    if fmt in {"polygon_normalized", "polygon", "poly", "quadrilateral", "xyxyxyxy_normalized", "roboflow_polygon"}:
        if len(parts) < 9:
            return None
        coords = [float(x) for x in parts[1:9]]
        xs = coords[0::2]
        ys = coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    elif fmt in {"xyxy_normalized", "xyxy", "corners", "pascal_normalized"}:
        x1, y1, x2, y2 = map(float, parts[1:5])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    elif fmt in {"yolo_xywh", "xywh", "ultralytics_xywh"}:
        xc, yc, w, h = map(float, parts[1:5])
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    else:
        raise ValueError(f"Unsupported label format: {box_format}")
    return x1, y1, x2, y2


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_yaml = Path(args.data)
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = data_yaml.parent.resolve()
    names = data.get("names", [])
    nc = int(data.get("nc", len(names) if isinstance(names, list) else len(names or [])))
    class_names = normalize_names(names, nc, args.config)

    print(f"data:        {args.data}")
    print(f"config:      {args.config}")
    print(f"box_format:  {cfg.labels.box_format}")
    print(f"nc:          {nc}")
    print("class mapping:")
    for i, name in enumerate(class_names):
        marker = ""
        if nc == 13 and i < len(DEFAULT_CHESS_CLASS_NAMES) and name != DEFAULT_CHESS_CLASS_NAMES[i]:
            marker = f"  WARNING expected lesson label: {DEFAULT_CHESS_CLASS_NAMES[i]!r}"
        print(f"  {i:2d}: {name!r}{marker}")

    train = collect_images(resolve_yaml_path(root, data.get("train"))) or collect_images([root / "train"])
    val = collect_images(resolve_yaml_path(root, data.get("val"))) or collect_images([root / "valid", root / "val"])
    test = collect_images(resolve_yaml_path(root, data.get("test"))) or collect_images([root / "test"])
    splits = {"train": train, "val": val, "valid": val, "test": test}
    paths = train + val + test if args.split == "all" else splits[args.split]

    counts = Counter()
    row_len_counts = Counter()
    missing = 0
    out_of_range = Counter()
    bad_rows = []
    degenerate = 0

    for img in paths:
        label = find_label_path(img)
        if not label.exists():
            missing += 1
            continue
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            row_len_counts[len(parts)] += 1
            try:
                cls = int(float(parts[0]))
                bbox = row_to_bbox(parts, cfg.labels.box_format)
            except Exception:
                bad_rows.append((str(label), line_no, line))
                continue
            if bbox is None:
                bad_rows.append((str(label), line_no, line))
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                degenerate += 1
            if 0 <= cls < nc:
                counts[cls] += 1
            else:
                out_of_range[cls] += 1

    print(f"images_checked:       {len(paths)}")
    print(f"missing_label_files:  {missing}")
    print(f"row_lengths:          {dict(sorted(row_len_counts.items()))}")
    print(f"bad_rows:             {len(bad_rows)}")
    print(f"degenerate_boxes:     {degenerate}")
    print("objects per class:")
    for i, name in enumerate(class_names):
        print(f"  {i:2d} {name}: {counts[i]}")

    if cfg.labels.box_format == "polygon_normalized" and row_len_counts and 9 not in row_len_counts:
        print("WARNING: config says polygon_normalized, but no 9-value rows were found.")
    if out_of_range:
        print("ERROR: out-of-range class ids found:", dict(out_of_range))
        raise SystemExit(2)
    if bad_rows:
        print("First bad rows:")
        for item in bad_rows[:10]:
            print(" ", item)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
