# -*- coding: utf-8 -*-
""" Module containing all the loss functions.

Writen by: Miquel Miró Nicolau (UIB)
"""
from typing import Callable, Union

import numpy as np

import tensorflow.keras.backend as keras
import tensorflow as tf


def onw_dice_coefficient(y_true, y_pred, smooth=1):
    """ Computes the Sørensen–Dice coefficient for the batch of images.

    Dice = (2*|X & Y|)/ (|X|+ |Y|)
        =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))

    Args:
        y_true: Batch of ground truth masks.
        y_pred: Batch of predicted masks.
        smooth: Smoothing factor. It adds to the numerator and denominator (intersection and union).

    Refs:
        https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = keras.sum(keras.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (
            keras.sum(keras.square(y_true), -1) + keras.sum(keras.square(y_pred), -1) + smooth)


def own_dice_coefficient_loss(y_true, y_pred):
    return 1 - onw_dice_coefficient(y_true, y_pred)


def multiclass_weighted_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[
    [tf.Tensor, tf.Tensor], tf.Tensor]:
    """ Weighted Dice loss.

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    Args:
        class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor],
                        len=<N_CLASSES>)
    Returns:
        Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Compute weighted Dice loss.

        Args:
            y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>,
                                                  <N_CLASSES>))
            y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>,
                                                       <N_CLASSES>))

        Returns:
            Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, keras.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * keras.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights  # Broadcasting
        denominator = keras.sum(denominator, axis=axis_to_reduce)

        iou = 1 - numerator / denominator

        return iou

    return loss


def dice_coef_loss(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5,
                   mean: bool = True):
    """ Dice coefficient loss function.

    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    From: TensorLayer

    Args:
        output (Tensor): A distribution with shape: [batch_size, ....], (any dimensions).
        target (Tensor): The target distribution, format the same with `output`.
        loss_type (str): ``jaccard`` or ``sorensen``, default is ``sorensen``.
        axis (tuple | int | None): All dimensions are reduced, default ``[1,2,3]``.
        smooth (float):  This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
            dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to
            0 (even the image values lower than the threshold), so in this case, higher smooth
            can have a higher dice.
        mean (bool):

    Returns:
    References:
        `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`
    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)

    if mean:
        dice = tf.reduce_mean(dice, name='dice_coeff')

    return 1 - dice


def positive_cce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Categorical cross entropy only for the slices with some positive element.

    To calculate the loss only takes into consideration the channels with some value different to 0
    in the ground truth.

    Args:
        y_true (tf.Tensor):
        y_pred (tf.Tensor):

    Returns:
        Positive CCE loss (tf.Tensor, shape=(None,))
    """
    zero_values = tf.reduce_sum(y_true, -1)
    zero_values = tf.reduce_sum(zero_values, -1)

    y_true = keras.reshape(y_true, (-1, y_true.shape[-2], y_true.shape[-1]))
    y_pred = keras.reshape(y_pred, (-1, y_true.shape[-2], y_true.shape[-1]))

    positives = keras.reshape(zero_values, (-1, 1))[:, 0]
    postives_ix = tf.compat.v1.where(positives > 0)[:, 0]

    y_true = tf.gather(y_true, postives_ix)
    y_pred = tf.gather(y_pred, postives_ix)

    loss = keras.switch(tf.size(input=y_true) > 0,
                        keras.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
    loss = keras.mean(loss)

    return loss


def smooth_l1_loss(y_true, y_pred):
    """ Implements Smooth-L1 loss.

    y_true and y_pred are typically: [N, 4], but could be any shape.

    Args:
        y_true: Ground truth bounding boxes, e.g. [None, 4] for [x1, y1, x2, y2]
        y_pred: Predicted bounding boxes, e.g. [None, 4] for [x1, y1, x2, y2]

    Returns:
        Smooth-L1 Loss value.
    """
    diff = keras.abs(y_true - y_pred)
    less_than_one = keras.cast(keras.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    return loss


def class_loss_graph(rpn_match, rpn_class_logits):
    """ RPN anchor classifier loss.

    Args:
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = keras.cast(keras.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.compat.v1.where(keras.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = keras.sparse_categorical_crossentropy(target=anchor_class,
                                                 output=rpn_class_logits,
                                                 from_logits=True)
    loss = keras.switch(tf.size(input=loss) > 0, keras.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """ Mask binary cross-entropy loss for the masks head.

    Args:
        target_masks [batch, num_rois, height, width]: A float32 tensor of values 0 or 1. Uses zero
                                                       padding to fill array.
        target_class_ids [batch, num_rois]: Integer class IDs. Zero padded.
        pred_masks [batch, height, width, num_classes]: float32 tensor with values from 0
                                                                   to 1.

    Returns:

    """
    pred_masks = tf.transpose(a=pred_masks, perm=[0, 3, 1, 2])
    target_masks = tf.transpose(a=target_masks, perm=[0, 3, 1, 2])

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = keras.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(input=target_masks)
    target_masks = keras.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_masks = keras.reshape(pred_masks, (-1, mask_shape[2], mask_shape[3]))

    # Only positive ROIs contribute to the loss. And only the class specific mask of each ROI.
    positive_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather(pred_masks, positive_ix)

    # Balanced the mask and the prediction with the same number of positive and negative samples
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))

    positive_px = tf.compat.v1.where(tf.math.greater(y_true, 0))

    negative_px = tf.compat.v1.where(tf.equal(y_true, 0))
    negative_px = tf.random.shuffle(negative_px)[:tf.size(positive_px)]

    y_true_pos = tf.gather(y_true, positive_px)
    y_pred_pos = tf.gather(y_pred, positive_px)

    y_true_neg = tf.gather(y_true, negative_px)
    y_pred_neg = tf.gather(y_pred, negative_px)

    y_true = tf.concat([y_true_pos, y_true_neg], axis=-1)
    y_pred = tf.concat([y_pred_pos, y_pred_neg], axis=-1)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]

    loss = keras.switch(tf.math.greater(tf.size(input=y_true), 0),
                        keras.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
    loss = keras.mean(loss)
    return loss


def bbox_loss_graph(target_bbox, rpn_match, rpn_bbox, batch_size: int = 3):
    """

    Args:
        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))]. Uses 0 padding to
                    fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=pos, -1=neg, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        batch_size

    Returns:

    """

    def batch_pack_graph(x, counts, num_rows):
        """ Picks different number of values from each row in x depending on the values in counts.

        TODO: Make num_rows (batch size) compatible with tf 2.7

        Args:
            x:
            counts:
            num_rows:

        Returns:

        """
        outputs = []
        for i in range(6):
            outputs.append(x[i, :counts[i]])  # I imatge, counts[i] bboxes
        return tf.concat(outputs, axis=0)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = keras.squeeze(rpn_match, -1)
    indices = tf.compat.v1.where(keras.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # print((target_bbox.shape, rpn_match.shape, rpn_bbox.shape))
    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = keras.sum(keras.cast(keras.equal(rpn_match, 1), tf.int32), axis=1)

    target_bbox = batch_pack_graph(target_bbox, batch_counts, batch_size)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = keras.switch(tf.size(input=loss) > 0, keras.mean(loss), tf.constant(0.0))

    return loss


def dice_coefficient_loss_rpn(target_masks, target_class_ids, pred_masks):
    """

    Args:
        target_masks: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1. Uses zero
                                                       padding to fill array.
        target_class_ids: Convention
        pred_masks: ([batch, num_rois, height, width]) float32 tensor with values from 0 to 1.

    Returns:

    """
    loss = keras.switch(tf.math.greater(tf.size(input=target_masks), 0),
                        own_dice_coefficient_loss(target_masks, pred_masks),
                        tf.constant(0.0))
    loss = keras.mean(loss)
    return loss
