from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy

from .model import yolo_boxes


def _broadcast_iou(box_1, box_2):
    """Tensor IoU for xyxy boxes.

    box_1: (..., 4)
    box_2: (N, 4)
    returns: (..., N)
    """
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.reshape(box_2, (1, 1, 1, 1, -1, 4))

    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h

    box_1_area = tf.maximum(box_1[..., 2] - box_1[..., 0], 0) * tf.maximum(box_1[..., 3] - box_1[..., 1], 0)
    box_2_area = tf.maximum(box_2[..., 2] - box_2[..., 0], 0) * tf.maximum(box_2[..., 3] - box_2[..., 1], 0)
    union_area = box_1_area + box_2_area - int_area

    return int_area / tf.maximum(union_area, 1e-10)


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    """YOLOv3 loss aligned with the lesson, but made safe for real training.

    The lesson shows the conceptual loss. This version keeps the same parts:
    xy_loss + wh_loss + obj_loss + class_loss, but uses tensor-safe IoU and
    handles images without objects in a batch.
    """
    anchors = tf.cast(anchors, tf.float32)

    def yolo_loss(y_true, y_pred):
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        grid = tf.cast(grid, tf.float32)

        true_xy = true_xy * tf.cast(grid_size, tf.float32) - grid
        true_wh = tf.math.log(tf.maximum(true_wh, 1e-10) / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
        true_wh = tf.where(tf.math.is_nan(true_wh), tf.zeros_like(true_wh), true_wh)

        obj_mask = tf.squeeze(true_obj, -1)
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))

        def build_ignore_mask():
            best_iou = tf.reduce_max(_broadcast_iou(pred_box, true_box_flat), axis=-1)
            return tf.cast(best_iou < ignore_thresh, tf.float32)

        ignore_mask = tf.cond(
            tf.shape(true_box_flat)[0] > 0,
            build_ignore_mask,
            lambda: tf.ones_like(obj_mask, dtype=tf.float32),
        )

        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        true_class_idx = tf.cast(tf.squeeze(true_class_idx, -1), tf.int32)
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
    """Lesson helper: convert padded boxes to a target tensor for one YOLO scale."""
    n = tf.shape(y_true)[0]
    y_true_out = tf.zeros((n, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6), dtype=tf.float32)

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0

    for i in tf.range(n):
        for j in tf.range(tf.shape(y_true)[1]):
            # y_true row is [x1, y1, x2, y2, class, best_anchor_idx]
            if tf.equal(y_true[i][j][2], 0):
                continue

            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)
                grid_xy = tf.clip_by_value(grid_xy, 0, grid_size - 1)

                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, classes):
    """Lesson helper: assign every box to best anchor and split targets by 13/26/52 grids."""
    outputs = []
    grid_size = 13

    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / tf.maximum(box_area + anchor_area - intersection, 1e-10)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        outputs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs, classes))
        grid_size *= 2

    return tuple(outputs)
