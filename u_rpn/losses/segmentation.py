# -*- coding: utf-8 -*-
""" Segmentation losses for the training of FCN models.

The losses defined in this module has as the main feature the ability to be used for the comparasion
between a ground truth and a predicted segmentation. All the data is supposed to be a tensor of at
least three dimensions. The first dimension is the height, the second is the width and the third is
is the channel.

Losses:
    dice_expanded: Dice loss without an aggregation operation. The results will be a bi-dimensional
        tensor with the same width and height than the input containing the error for each "voxel".
    dice_multi_class_weighted: Wrapper for the dice loss with an aggregation operation and a weight
        for each class.
    dice_aggregated: Dice loss with an aggregation operation. The results will be a single value for
        each channel.
    dice_rpn: Dice loss adapted to the U-RPN architecture.
    mrcnn_mask_loss_graph: Loss for the mask prediction of the Mask R-CNN model. Based on the
        binary cross entropy loss and applied only to the positives channels.

Writen by: Miquel Miró Nicolau (UIB)
"""
from typing import Union, Callable

import numpy as np
import tensorflow.keras.backend as keras
import tensorflow as tf


def dice_expanded(y_true, y_pred, smooth=1):
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
    dice = (2. * intersection + smooth) / (
            keras.sum(keras.square(y_true), -1) + keras.sum(keras.square(y_pred), -1) + smooth)

    return 1 - dice


def dice_multiclass_weighted(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[
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


def dice_aggregated_loss(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5,
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


def dice_rpn(target_masks, target_class_ids, pred_masks):
    """

    Args:
        target_masks: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1. Uses zero
                                                       padding to fill array.
        target_class_ids: Convention
        pred_masks: ([batch, num_rois, height, width]) float32 tensor with values from 0 to 1.

    Returns:

    """
    loss = keras.switch(tf.math.greater(tf.size(input=target_masks), 0),
                        dice_expanded(target_masks, pred_masks),
                        tf.constant(0.0))
    loss = keras.mean(loss)
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
