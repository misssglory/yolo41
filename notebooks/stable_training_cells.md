# Stable notebook training cells

Use these values after verifying polygon ground truth labels:

```python
from yolo_chess.notebook_api import train_notebook

result = train_notebook(
    config_path=CONFIG_PATH,
    data_yaml=DATA_YAML,
    epochs=20,
    batch=2,
    lr=1e-5,
    fine_tune_lr=3e-6,
    clipnorm=1.0,
    freeze_darknet_epochs=3,
    device="auto",
    weights="none",
    pretrained_darknet=True,
    out="runs/detect/yolov3_keras_chess",
    no_increment=False,
)

BEST_WEIGHTS = result["best_weights"]
CLASS_NAMES = result["dataset_info"].class_names
NUM_CLASSES = result["dataset_info"].num_classes
```

If predictions become `[0, 0, 415, 415]` with score `1.000`, delete that run and restart from pretrained weights with a lower learning rate.
