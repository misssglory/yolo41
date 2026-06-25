from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from .config import ANCHORS, ANCHOR_MASKS, SIZE, load_config
from .dataset import load_dataset_info, make_dataset
from .download import ensure_data_yaml
from .losses import YoloLoss
from .model import YoloV3, load_darknet_weights_for_finetune


def make_safe_optimizer(lr: float, clipnorm: float | None = 1.0) -> tf.keras.optimizers.Optimizer:
    """Adam with optional gradient clipping.

    For YOLOv3, clipping is important because width/height are decoded with
    exp(tw/th); one oversized update can saturate boxes to the whole image.
    """
    if clipnorm is None or clipnorm <= 0:
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)


def set_darknet_trainable(model: tf.keras.Model, trainable: bool) -> None:
    """Freeze/unfreeze the Darknet backbone while keeping YOLO heads trainable."""
    try:
        model.get_layer("yolo_darknet").trainable = trainable
    except ValueError:
        pass


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for hist in histories:
        if hist is None:
            continue
        for key, values in hist.history.items():
            merged.setdefault(key, []).extend(list(values))
    return merged


from .train import (
    choose_device,
    ensure_darknet_weights,
    increment_run_dir,
    load_weights_safely,
    save_history_csv,
    save_loss_plot,
)


def train_notebook(
    config_path: str | Path = "config.toml",
    data_yaml: str | Path = "chess_yolo/data.yaml",
    epochs: int = 15,
    batch: int = 2,
    lr: float | None = None,
    clipnorm: float | None = None,
    freeze_darknet_epochs: int | None = None,
    fine_tune_lr: float | None = None,
    device: str = "auto",
    out: str | Path = "runs/detect/yolov3_keras_chess",
    weights: str | None = "none",
    pretrained_darknet: bool | None = None,
    darknet_weights: str | None = None,
    darknet_weights_url: str | None = None,
    no_increment: bool = False,
    download_if_missing: bool = True,
) -> dict[str, Any]:
    """Train YOLOv3 from notebook without subprocess.

    This is the same flow as `python -m yolo_chess.train`, but callable directly
    from an ipynb cell.
    """
    cfg = load_config(config_path)
    lr = cfg.training.learning_rate if lr is None else lr
    clipnorm = cfg.training.clipnorm if clipnorm is None else clipnorm
    freeze_darknet_epochs = cfg.training.freeze_darknet_epochs if freeze_darknet_epochs is None else freeze_darknet_epochs
    fine_tune_lr = cfg.training.fine_tune_learning_rate if fine_tune_lr is None else fine_tune_lr
    weights_mode = weights if weights is not None else cfg.training.weights
    use_darknet = cfg.training.use_darknet_pretrained if pretrained_darknet is None else pretrained_darknet
    darknet_weights = darknet_weights or cfg.training.darknet_weights
    darknet_weights_url = darknet_weights_url or cfg.training.darknet_weights_url

    data_yaml = ensure_data_yaml(data_yaml, download_if_missing=download_if_missing)
    info = load_dataset_info(data_yaml, config_path=config_path)

    if not info.train_images:
        raise RuntimeError(f"No training images found from {data_yaml}")

    out_dir = Path(out)
    if not no_increment:
        out_dir = increment_run_dir(out_dir)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset:")
    print(f"  root:    {info.root}")
    print(f"  train:   {len(info.train_images)} images")
    print(f"  val:     {len(info.val_images)} images")
    print(f"  test:    {len(info.test_images)} images")
    print(f"  classes: {info.num_classes} -> {info.class_names}")
    print(f"  run dir: {out_dir}")
    print("Labels:")
    print(f"  box_format: {cfg.labels.box_format}")
    print("Training stability:")
    print(f"  lr:                    {lr}")
    print(f"  fine_tune_lr:          {fine_tune_lr}")
    print(f"  clipnorm:              {clipnorm}")
    print(f"  freeze_darknet_epochs: {freeze_darknet_epochs}")

    train_ds = make_dataset(info.train_images, batch, info.num_classes, shuffle=True, config_path=config_path)
    val_ds = make_dataset(info.val_images, batch, info.num_classes, shuffle=False, config_path=config_path)

    logical_device = choose_device(device)
    print(f"Using TensorFlow device: {logical_device}")

    with tf.device(logical_device):
        model = YoloV3(size=SIZE, classes=info.num_classes, training=True)
        loaded = load_weights_safely(model, str(weights_mode), Path("runs/detect"))

        darknet_loaded = None
        if loaded is None and use_darknet:
            try:
                darknet_path = ensure_darknet_weights(darknet_weights, darknet_weights_url)
                darknet_loaded = load_darknet_weights_for_finetune(
                    model,
                    str(darknet_path),
                    classes=info.num_classes,
                    darknet_classes=80,
                )
                print(f"Loaded Darknet pretrained weights for fine-tuning: {darknet_path}")
                print(f"Darknet load summary: {darknet_loaded}")
            except Exception as exc:
                print("WARNING: could not load Darknet pretrained weights; continuing with random initialization.")
                print(f"Reason: {type(exc).__name__}: {exc}")

        losses = [
            YoloLoss(ANCHORS[ANCHOR_MASKS[0]], classes=info.num_classes),
            YoloLoss(ANCHORS[ANCHOR_MASKS[1]], classes=info.num_classes),
            YoloLoss(ANCHORS[ANCHOR_MASKS[2]], classes=info.num_classes),
        ]

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
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ]

        histories = []
        freeze_darknet_epochs = max(0, min(int(freeze_darknet_epochs), int(epochs)))

        if freeze_darknet_epochs > 0:
            print(f"Phase 1/{2 if freeze_darknet_epochs < epochs else 1}: train YOLO heads only for {freeze_darknet_epochs} epochs")
            set_darknet_trainable(model, False)
            model.compile(optimizer=make_safe_optimizer(lr, clipnorm), loss=losses)
            histories.append(
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=freeze_darknet_epochs,
                    callbacks=callbacks,
                )
            )

        if freeze_darknet_epochs < epochs:
            print(f"Phase 2/2: fine-tune all layers from epoch {freeze_darknet_epochs + 1} to {epochs}")
            set_darknet_trainable(model, True)
            model.compile(optimizer=make_safe_optimizer(fine_tune_lr, clipnorm), loss=losses)
            histories.append(
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    initial_epoch=freeze_darknet_epochs,
                    epochs=epochs,
                    callbacks=callbacks,
                )
            )

        history = tf.keras.callbacks.History()
        history.history = merge_histories(*histories)

    model.save_weights(str(weights_dir / "final.weights.h5"))
    save_history_csv(history, out_dir / "history.csv")
    save_loss_plot(history, out_dir / "loss.png")
    (out_dir / "classes.txt").write_text("\n".join(info.class_names), encoding="utf-8")
    (out_dir / "training_summary.txt").write_text(
        "\n".join(
            [
                "YOLOv3 Keras chess training",
                f"data={data_yaml}",
                f"classes={info.num_classes}",
                f"class_names={info.class_names}",
                f"label_box_format={cfg.labels.box_format}",
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

    return {
        "model": model,
        "history": history,
        "dataset_info": info,
        "out_dir": out_dir,
        "weights_dir": weights_dir,
        "best_weights": weights_dir / "best.weights.h5",
        "last_weights": weights_dir / "last.weights.h5",
        "final_weights": weights_dir / "final.weights.h5",
    }
