from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from .config import ANCHORS, ANCHOR_MASKS, SIZE, load_config
from .dataset import load_dataset_info, make_dataset
from .download import ensure_data_yaml
from .losses import YoloLoss
from .model import YoloV3, load_darknet_weights_for_finetune


def choose_device(device: str) -> str:
    if device == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        return "/GPU:0" if gpus else "/CPU:0"
    if device == "cpu":
        return "/CPU:0"
    if device.startswith("gpu"):
        idx = device.replace("gpu", "").replace(":", "") or "0"
        return f"/GPU:{idx}"
    return device


def increment_run_dir(base: Path) -> Path:
    if not base.exists():
        return base
    parent = base.parent
    stem = base.name
    idx = 2
    while True:
        candidate = parent / f"{stem}{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def find_latest_weights(runs_root: Path) -> Path | None:
    candidates = []
    for pattern in ("yolov3_keras_chess*/weights/last.weights.h5", "yolov3_keras_chess*/weights/best.weights.h5"):
        candidates.extend(runs_root.glob(pattern))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_darknet_weights(path: str | Path, url: str) -> Path:
    """Download original YOLOv3 Darknet weights from the lesson if missing."""
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Darknet weights not found: {path}")
    print(f"Downloading YOLOv3 COCO weights: {url}")
    urllib.request.urlretrieve(url, path)
    return path


def load_weights_safely(model: tf.keras.Model, weights: str, runs_root: Path) -> Path | None:
    if weights == "none":
        print("Training from scratch: --weights none")
        return None

    weights_path: Path | None
    if weights == "auto":
        weights_path = find_latest_weights(runs_root)
        if weights_path is None:
            print("No previous weights found. Training from scratch.")
            return None
    else:
        weights_path = Path(weights)
        if not weights_path.exists():
            print(f"WARNING: weights file not found: {weights_path}. Training from scratch.")
            return None

    try:
        model.load_weights(str(weights_path))
        print(f"Loaded existing model weights: {weights_path}")
        return weights_path
    except Exception as exc:
        print("WARNING: could not load weights. Training from scratch.")
        print(f"Weights path: {weights_path}")
        print(f"Reason: {type(exc).__name__}: {exc}")
        return None


def save_history_csv(history: tf.keras.callbacks.History, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for epoch_idx in range(len(history.history[keys[0]])):
            writer.writerow([epoch_idx + 1, *[history.history[k][epoch_idx] for k in keys]])


def save_loss_plot(history: tf.keras.callbacks.History, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("YOLOv3 chess training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv3 from the lesson on chess pieces.")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--data", default="chess_yolo/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, gpu0, /GPU:0, /CPU:0")
    parser.add_argument("--out", default="runs/detect/yolov3_keras_chess")
    parser.add_argument("--weights", default=None, help="auto | none | path/to/*.weights.h5. If omitted, value comes from [training].weights in config.toml")
    parser.add_argument("--pretrained-darknet", action=argparse.BooleanOptionalAction, default=None, help="Load original YOLOv3 COCO Darknet weights before fine-tuning if no local Keras weights were loaded")
    parser.add_argument("--darknet-weights", default=None, help="Path to original yolov3.weights. If omitted, value comes from config.toml")
    parser.add_argument("--darknet-weights-url", default=None, help="URL for downloading yolov3.weights if missing")
    parser.add_argument("--no-increment", action="store_true", help="Write into --out directly instead of creating yolov3_keras_chess2 etc.")
    parser.add_argument("--download-if-missing", action=argparse.BooleanOptionalAction, default=True, help="Download chess_yolo.zip automatically if default chess_yolo/data.yaml is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    weights_mode = args.weights if args.weights is not None else cfg.training.weights
    pretrained_darknet = (
        args.pretrained_darknet
        if args.pretrained_darknet is not None
        else cfg.training.use_darknet_pretrained
    )
    darknet_weights = args.darknet_weights or cfg.training.darknet_weights
    darknet_weights_url = args.darknet_weights_url or cfg.training.darknet_weights_url

    data_yaml = ensure_data_yaml(args.data, download_if_missing=args.download_if_missing)
    dataset_info = load_dataset_info(data_yaml, config_path=args.config)

    if not dataset_info.train_images:
        raise RuntimeError(f"No training images found from {args.data}")

    out_dir = Path(args.out)
    if not args.no_increment:
        out_dir = increment_run_dir(out_dir)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset:")
    print(f"  root:    {dataset_info.root}")
    print(f"  train:   {len(dataset_info.train_images)} images")
    print(f"  val:     {len(dataset_info.val_images)} images")
    print(f"  test:    {len(dataset_info.test_images)} images")
    print(f"  classes: {dataset_info.num_classes} -> {dataset_info.class_names}")
    print(f"  run dir: {out_dir}")
    print("Initialization:")
    print(f"  existing weights:      {weights_mode}")
    print(f"  pretrained darknet:   {pretrained_darknet}")
    print(f"  darknet weights path: {darknet_weights}")

    train_ds = make_dataset(dataset_info.train_images, args.batch, dataset_info.num_classes, shuffle=True)
    val_ds = make_dataset(dataset_info.val_images, args.batch, dataset_info.num_classes, shuffle=False)

    logical_device = choose_device(args.device)
    print(f"Using TensorFlow device: {logical_device}")

    with tf.device(logical_device):
        model = YoloV3(size=SIZE, classes=dataset_info.num_classes, training=True)
        loaded = load_weights_safely(model, weights_mode, Path("runs/detect"))

        darknet_loaded = None
        if loaded is None and pretrained_darknet:
            try:
                darknet_path = ensure_darknet_weights(darknet_weights, darknet_weights_url)
                darknet_loaded = load_darknet_weights_for_finetune(
                    model,
                    str(darknet_path),
                    classes=dataset_info.num_classes,
                    darknet_classes=80,
                )
                print(f"Loaded Darknet pretrained weights for fine-tuning: {darknet_path}")
                print(f"Darknet load summary: {darknet_loaded}")
            except Exception as exc:
                print("WARNING: could not load Darknet pretrained weights; continuing with random initialization.")
                print(f"Reason: {type(exc).__name__}: {exc}")

        losses = [
            YoloLoss(ANCHORS[ANCHOR_MASKS[0]], classes=dataset_info.num_classes),
            YoloLoss(ANCHORS[ANCHOR_MASKS[1]], classes=dataset_info.num_classes),
            YoloLoss(ANCHORS[ANCHOR_MASKS[2]], classes=dataset_info.num_classes),
        ]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=losses)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(weights_dir / "last.weights.h5"),
                save_weights_only=True,
                save_best_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(weights_dir / "best.weights.h5"),
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "keras_log.csv")),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
        )

    # Save final weights too, so even interrupted ModelCheckpoint naming is not a mystery.
    model.save_weights(str(weights_dir / "final.weights.h5"))
    save_history_csv(history, out_dir / "history.csv")
    save_loss_plot(history, out_dir / "loss.png")

    (out_dir / "classes.txt").write_text("\n".join(dataset_info.class_names), encoding="utf-8")
    (out_dir / "training_summary.txt").write_text(
        "\n".join(
            [
                "YOLOv3 Keras chess training",
                f"data={data_yaml}",
                f"classes={dataset_info.num_classes}",
                f"class_names={dataset_info.class_names}",
                f"loaded_weights={loaded}",
                f"loaded_darknet_pretrained={darknet_loaded}",
                f"image_size={SIZE}",
                f"anchors={ANCHORS.tolist()}",
                f"anchor_masks={ANCHOR_MASKS.tolist()}",
            ]
        ),
        encoding="utf-8",
    )

    print("Training complete.")
    print(f"Best weights:  {weights_dir / 'best.weights.h5'}")
    print(f"Last weights:  {weights_dir / 'last.weights.h5'}")
    print(f"Final weights: {weights_dir / 'final.weights.h5'}")


if __name__ == "__main__":
    main()
