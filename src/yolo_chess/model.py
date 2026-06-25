from __future__ import annotations

from itertools import repeat

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2

from .config import ANCHORS, ANCHOR_MASKS, YOLO_IOU_THRESHOLD, YOLO_SCORE_THRESHOLD


def DBL(x, filters, kernel, strides=1, batch_norm=True):
    """Darknet Block Layer from the lesson: Conv2D + optional BN + LeakyReLU."""
    if strides == 1:
        padding = "same"
    else:
        # Same idea as in the lesson: pad top/left before stride-2 convolution.
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"

    x = Conv2D(
        filters=filters,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=l2(0.0005),
    )(x)

    if batch_norm:
        x = BatchNormalization(epsilon=0.001)(x)
        x = LeakyReLU(negative_slope=0.1)(x)
    return x


def ResUnit(x, filters):
    """Residual unit from the lesson."""
    skip_connection = x
    x = DBL(x, filters // 2, 1)
    x = DBL(x, filters, 3)
    x = Add()([skip_connection, x])
    return x


def ResN(x, filters, blocks):
    """Residual block group from the lesson."""
    x = DBL(x, filters, kernel=3, strides=2)
    for _ in repeat(None, blocks):
        x = ResUnit(x, filters)
    return x


def Darknet(name=None):
    """Darknet-53 backbone with three routes: 52x52, 26x26, 13x13 for input 416."""
    x = inputs = Input([None, None, 3])
    x = DBL(x, 32, 3)
    x = ResN(x, 64, 1)
    x = ResN(x, 128, 2)
    x = route_1 = ResN(x, 256, 8)
    x = route_2 = ResN(x, 512, 8)
    route_3 = ResN(x, 1024, 4)
    return tf.keras.Model(inputs, (route_1, route_2, route_3), name=name)


def YoloHead(filters, name=None):
    """YOLOv3 head from the lesson.

    Can accept either one tensor or a tuple: (previous_deep_head, route_skip).
    """

    def layer(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            x = DBL(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DBL(x, filters, 1)
        x = DBL(x, filters * 2, 3)
        x = DBL(x, filters, 1)
        x = DBL(x, filters * 2, 3)
        x = DBL(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return layer


def YoloHeadOutput(filters, anchors, classes, name=None):
    """YOLO output head from the lesson: DBL + Conv + reshape to SxSx3x(classes+5)."""

    def layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DBL(x, filters * 2, 3)
        x = DBL(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(
            lambda t: tf.reshape(
                t,
                (-1, tf.shape(t)[1], tf.shape(t)[2], anchors, classes + 5),
            )
        )(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return layer


def yolo_boxes(pred, anchors, classes):
    """Decode raw YOLO predictions to normalized xyxy boxes.

    This follows the equations shown in the lesson:
    bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy,
    bw = pw * exp(tw), bh = ph * exp(th).
    """
    grid_size = tf.shape(pred)[1]

    box_xy, box_wh, score, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    score = tf.sigmoid(score)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    grid = tf.cast(grid, tf.float32)

    anchors = tf.cast(anchors, tf.float32)
    b_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    b_wh = tf.exp(box_wh) * anchors

    box_x1y1 = b_xy - b_wh / 2
    box_x2y2 = b_xy + b_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, score, class_probs, pred_box


def xyxy_to_yxyx(boxes):
    """Convert [x1, y1, x2, y2] boxes to TensorFlow NMS format [y1, x1, y2, x2]."""
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    return tf.concat([y1, x1, y2, x2], axis=-1)


def yxyx_to_xyxy(boxes):
    """Convert TensorFlow NMS boxes [y1, x1, y2, x2] back to project format [x1, y1, x2, y2]."""
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
    return tf.concat([x1, y1, x2, y2], axis=-1)


def nonMaximumSuppression(outputs, anchors, masks, classes):
    """NMS aligned with the lesson, with explicit coordinate-order conversion.

    yolo_boxes returns [x1, y1, x2, y2].
    tf.image.combined_non_max_suppression expects [y1, x1, y2, x2].
    Returning boxes are converted back to [x1, y1, x2, y2] so the rest of
    the project and notebook cells can draw them correctly.
    """
    boxes, conf, out_type = [], [], []

    for output in outputs:
        boxes.append(tf.reshape(output[0], (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])))
        conf.append(tf.reshape(output[1], (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])))
        out_type.append(tf.reshape(output[2], (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])))

    bbox_xyxy = tf.concat(boxes, axis=1)
    confidence = tf.concat(conf, axis=1)
    class_probs = tf.concat(out_type, axis=1)
    scores = confidence * class_probs

    bbox_yxyx = xyxy_to_yxyx(bbox_xyxy)

    nms_boxes_yxyx, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox_yxyx, (tf.shape(bbox_yxyx)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=YOLO_IOU_THRESHOLD,
        score_threshold=YOLO_SCORE_THRESHOLD,
    )

    nms_boxes_xyxy = yxyx_to_xyxy(nms_boxes_yxyx)
    return nms_boxes_xyxy, nms_scores, nms_classes, valid_detections


def YoloV3(
    size=None,
    channels=3,
    anchors=ANCHORS,
    masks=ANCHOR_MASKS,
    classes=80,
    training=False,
):
    """YOLOv3 model builder, structurally aligned with the lesson.

    training=True  -> returns raw outputs for loss.
    training=False -> returns post-processed boxes/scores/classes/valid_detections.
    """
    x = inputs = Input([size, size, channels])

    route_1, route_2, route_3 = Darknet(name="yolo_darknet")(x)

    x = YoloHead(512, name="yolo_head_1")(route_3)
    output_0 = YoloHeadOutput(512, len(masks[0]), classes, name="yolo_output_1")(x)

    x = YoloHead(256, name="yolo_head_2")((x, route_2))
    output_1 = YoloHeadOutput(256, len(masks[1]), classes, name="yolo_output_2")(x)

    x = YoloHead(128, name="yolo_head_3")((x, route_1))
    output_2 = YoloHeadOutput(128, len(masks[2]), classes, name="yolo_output_3")(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name="yolov3")

    boxes_0 = Lambda(lambda t: yolo_boxes(t, anchors[masks[0]], classes), name="yolo_boxes_0")(output_0)
    boxes_1 = Lambda(lambda t: yolo_boxes(t, anchors[masks[1]], classes), name="yolo_boxes_1")(output_1)
    boxes_2 = Lambda(lambda t: yolo_boxes(t, anchors[masks[2]], classes), name="yolo_boxes_2")(output_2)

    outputs = Lambda(
        lambda t: nonMaximumSuppression(t, anchors, masks, classes),
        name="nonMaximumSuppression",
    )((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name="yolov3")


YOLO_V3_LAYERS = [
    "yolo_darknet",
    "yolo_head_1",
    "yolo_output_1",
    "yolo_head_2",
    "yolo_output_2",
    "yolo_head_3",
    "yolo_output_3",
]


def _next_batch_norm_layer(layers, idx):
    """Return the BatchNormalization layer after a Conv2D, if it exists."""
    if idx + 1 >= len(layers):
        return None
    candidate = layers[idx + 1]
    return candidate if isinstance(candidate, BatchNormalization) else None


def load_darknet_weights_for_finetune(
    model: tf.keras.Model,
    weights_file: str,
    classes: int,
    darknet_classes: int = 80,
    layers: list[str] | None = None,
) -> dict[str, int]:
    """Load original YOLOv3 Darknet weights into the lesson model before fine-tuning.

    The lesson downloads COCO weights for 80 classes.  For the chess homework we build
    the same YOLOv3 architecture, but the final output convolutions have
    3 * (classes + 5) filters instead of 3 * (80 + 5) filters.

    Therefore this loader reads the whole .weights file, loads all compatible
    backbone/head layers, and deliberately skips only the 3 final output conv layers
    whose shapes do not match the chess class count. Those skipped layers remain
    randomly initialized and are trained from scratch.
    """
    layers = layers or YOLO_V3_LAYERS
    darknet_output_filters = 3 * (darknet_classes + 5)

    loaded_conv = 0
    skipped_conv = 0
    loaded_bn = 0

    with open(weights_file, "rb") as wf:
        # Darknet header: major, minor, revision, seen, _
        _ = np.fromfile(wf, dtype=np.int32, count=5)

        for layer_name in layers:
            sub_model = model.get_layer(layer_name)
            sub_layers = list(sub_model.layers)

            for i, layer in enumerate(sub_layers):
                if not isinstance(layer, Conv2D):
                    continue

                batch_norm = _next_batch_norm_layer(sub_layers, i)
                kernel_h = layer.kernel_size[0]
                kernel_w = layer.kernel_size[1]
                if kernel_h != kernel_w:
                    raise ValueError(f"Only square Conv2D kernels are expected, got {layer.kernel_size}")
                kernel_size = kernel_h
                in_dim = layer.get_weights()[0].shape[2]

                # The three final YOLO output convs have no BN and differ when classes != 80.
                is_output_conv_mismatch = batch_norm is None and layer.filters != darknet_output_filters
                file_filters = darknet_output_filters if is_output_conv_mismatch else layer.filters

                if batch_norm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=file_filters)
                else:
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * file_filters)
                    bn_weights = bn_weights.reshape((4, file_filters))[[1, 0, 2, 3]]

                conv_shape_darknet = (file_filters, in_dim, kernel_size, kernel_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=int(np.prod(conv_shape_darknet)))
                conv_weights = conv_weights.reshape(conv_shape_darknet).transpose([2, 3, 1, 0])

                expected_conv_shape = layer.get_weights()[0].shape
                if conv_weights.shape != expected_conv_shape:
                    # This is expected for the 3 final output layers when classes=13.
                    skipped_conv += 1
                    continue

                if batch_norm is None:
                    expected_bias_shape = layer.get_weights()[1].shape
                    if conv_bias.shape != expected_bias_shape:
                        skipped_conv += 1
                        continue
                    layer.set_weights([conv_weights, conv_bias])
                else:
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)
                    loaded_bn += 1
                loaded_conv += 1

        unread = wf.read()
        if unread:
            raise RuntimeError(f"Darknet weights were not fully consumed: {len(unread)} trailing bytes")

    return {
        "loaded_conv": loaded_conv,
        "loaded_batch_norm": loaded_bn,
        "skipped_conv": skipped_conv,
    }
