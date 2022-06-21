# -*- coding: utf-8 -*-
""" Segmentation losses for the training of FCN models.

The losses defined in this module has as the main feature the ability to be used for the comparasion
between a ground truth and a predicted segmentation. All the data is supposed to be a tensor of at
least three dimensions. The first dimension is the height, the second is the width and the third is
is the channel.

Losses:
    dice_expanded: Dice loss without an aggregation operation. The results will be a bi-dimensional
        tensor with the same width and height than the input containing the error for each "voxel".
    dice_aggregated: Dice loss with an aggregation operation. The results will be a single value for
        each channel.
    dice_rpn: Dice loss adapted to the U-RPN architecture.
    mrcnn_mask_loss_graph: Loss for the mask prediction of the Mask R-CNN model. Based on the
        binary cross entropy loss and applied only to the positives channels.

Writen by: Miquel Miró Nicolau (UIB)
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def dice_expanded(target: tf.Tensor, pred: tf.Tensor, smooth: int = 1):
    """Computes the Sørensen–Dice coefficient for the batch of images.

    Dice = (2*|X & Y|)/ (|X|+ |Y|)
        =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))

    Args:
        target: Batch of ground truth masks.
        pred: Batch of predicted masks.
        smooth: Smoothing factor. It adds to the numerator and denominator (intersection and union).

    Refs:
        https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(target * pred), axis=-1)
    dice = (2.0 * intersection + smooth) / (
        K.sum(K.square(target), -1) + K.sum(K.square(pred), -1) + smooth
    )

    return 1 - dice


def dice_aggregated_loss(
    target: tf.Tensor,
    pred: tf.Tensor,
    loss_type: str = "sorensen",
    axis: tuple = (1, 2, 3),
    smooth: float = 1e-5,
    mean: bool = True,
) -> tf.Tensor:
    """Dice coefficient loss function.

    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    From: TensorLayer

    Args:
        pred (Tensor): A distribution with shape: [batch_size, ....], (any dimensions).
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
    inse = tf.reduce_sum(pred * target, axis=axis)
    if loss_type == "jaccard":
        pred = tf.reduce_sum(pred * pred, axis=axis)
        target = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == "sorensen":
        pred = tf.reduce_sum(pred, axis=axis)
        target = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2.0 * inse + smooth) / (pred + target + smooth)

    if mean:
        dice = tf.reduce_mean(dice, name="dice_coeff")

    return 1 - dice


def dice_rpn(target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict):
    """

    Args:
        target: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1. Uses
                                    zero padding to fill array.
        pred: ([batch, num_rois, height, width]) float32 tensor with values from 0 to 1.
    Returns:

    """
    loss = K.switch(
        tf.math.greater(tf.size(input=target), 0),
        dice_expanded(target, pred),
        tf.constant(0.0),
    )
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(
    target_masks: np.ndarray, pred: np.ndarray, target_class_ids: np.ndarray
) -> tf.Tensor:
    """Mask binary cross-entropy loss for the masks head.

    Args:
        target_masks [batch, num_rois, height, width]: A float32 tensor of values 0 or 1. Uses zero
                                                       padding to fill array.
        target_class_ids [batch, num_rois]: Integer class IDs. Zero padded.
        pred [batch, height, width, num_classes]: float32 tensor with values from 0
                                                                   to 1.

    Returns:

    """
    pred = tf.transpose(a=pred, perm=[0, 3, 1, 2])
    target_masks = tf.transpose(a=target_masks, perm=[0, 3, 1, 2])

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(input=target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred = K.reshape(pred, (-1, mask_shape[2], mask_shape[3]))

    # Only positive ROIs contribute to the loss. And only the class specific mask of each ROI.
    positive_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather(pred, positive_ix)

    # Balanced the mask and the prediction with the same number of positive and negative samples
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))

    positive_px = tf.compat.v1.where(tf.math.greater(y_true, 0))

    negative_px = tf.compat.v1.where(tf.equal(y_true, 0))
    negative_px = tf.random.shuffle(negative_px)[: tf.size(positive_px)]

    y_true_pos = tf.gather(y_true, positive_px)
    y_pred_pos = tf.gather(y_pred, positive_px)

    y_true_neg = tf.gather(y_true, negative_px)
    y_pred_neg = tf.gather(y_pred, negative_px)

    y_true = tf.concat([y_true_pos, y_true_neg], axis=-1)
    y_pred = tf.concat([y_pred_pos, y_pred_neg], axis=-1)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]

    loss = K.switch(
        tf.math.greater(tf.size(input=y_true), 0),
        K.binary_crossentropy(target=y_true, output=y_pred),
        tf.constant(0.0),
    )
    loss = K.mean(loss)

    return loss
