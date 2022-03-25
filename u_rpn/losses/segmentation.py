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
    bce_weighted_rpn: Weighted BCE loss adapted to the U-RPN architecture.
    mrcnn_mask_loss_graph: Loss for the mask prediction of the Mask R-CNN model. Based on the
        binary cross entropy loss and applied only to the positives channels.

Writen by: Miquel Miró Nicolau (UIB)
"""
from typing import Optional

from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K


def dice_expanded(target, pred, smooth=1):
    """ Computes the Sørensen–Dice coefficient for the batch of images.

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
    dice = (2. * intersection + smooth) / (
            K.sum(K.square(target), -1) + K.sum(K.square(pred), -1) + smooth)

    return 1 - dice


def dice_aggregated_loss(target, pred, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5,
                         mean: bool = True):
    """ Dice coefficient loss function.

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
    if loss_type == 'jaccard':
        l = tf.reduce_sum(pred * pred, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(pred, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)

    if mean:
        dice = tf.reduce_mean(dice, name='dice_coeff')

    return 1 - dice


def dice_rpn(target, pred, *args, **kwargs):
    """

    Args:
        target: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1. Uses
                                    zero padding to fill array.
        pred: ([batch, num_rois, height, width]) float32 tensor with values from 0 to 1.
    Returns:

    """
    loss = K.switch(tf.math.greater(tf.size(input=target), 0),
                    dice_expanded(target, pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


class WeightedBCE(keras.losses.Loss):
    """ Weighted binary cross entropy loss for the RPN.

    To fix unbalanced classes, we split the loss function into positive and negative classes, and
    we weight the loss of the positive class with the number of positive examples in the batch and
    the same with the negative class.
    """

    def __init__(self, positive_weight: float = 0.5):
        super().__init__()
        self.__positive_w = positive_weight

    def call(self, target, pred, *args, **kwargs):
        """ Weighted binary cross entropy loss for the RPN.

        To fix unbalanced classes, we split the loss function into positive and negative classes,
        and we weight the loss of the positive class with the number of positive examples in the
        batch and the same with the negative class.

        Args:
            target: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1. Uses
                                zero padding to fill array.
            pred: ([batch, num_rois, height, width]) float32 tensor with values from 0 to 1.

        Returns:

        """
        pred = tf.reshape(pred, [-1])
        target = tf.reshape(target, [-1])

        pos_px = tf.compat.v1.where(tf.math.greater(target, 0))
        neg_px = tf.compat.v1.where(tf.math.less_equal(target, 0))

        pos_pred = tf.squeeze(tf.gather(pred, pos_px))
        pos_target = tf.squeeze(tf.gather(target, pos_px))

        neg_pred = tf.squeeze(tf.gather(pred, neg_px))
        neg_target = tf.squeeze(tf.gather(target, neg_px))

        bce = OwnBCE(reduction=tf.keras.losses.Reduction.SUM)

        pos_loss = bce(pos_target, pos_pred)
        neg_loss = bce(neg_target, neg_pred)

        return (tf.cast(neg_loss, tf.float32) * self.__positive_w) + (
                tf.cast(pos_loss, tf.float32) * (1 - self.__positive_w))


class WeightedTernaryBCE(keras.losses.Loss):
    """ Weighted ternary binary cross entropy loss for the RPN.

    To fix unbalanced classes, we split the loss function into positive and negative classes, and
    we weight the loss of the positive class with the number of positive examples in the batch and
    the same with the negative class.
    """

    def __init__(self, weights: Optional[float] = None):
        super().__init__()

        if weights is None:
            weights = [0.5, 0.5, 0.5]
        self.__weights = weights

    def call(self, target, pred, *args, **kwargs):
        """ Ternary weighted binary cross entropy loss for the RPN.

        To fix unbalanced classes, we split the loss function into positive, false negative and true
        negatives classes. Loss functions are calculated for each of these three cases and then
        merged.

        Args:
            target: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1. Uses
                                zero padding to fill array.
            pred: ([batch, num_rois, height, width]) float32 tensor with values from 0 to 1.

        Returns:

        """
        pred = tf.reshape(pred, [-1])
        target = tf.reshape(target, [-1])

        tp_px = tf.compat.v1.where(tf.math.greater(target, 0))
        neg_px = tf.compat.v1.where(tf.math.less_equal(target, 0))

        pos_pred = tf.squeeze(tf.gather(pred, tp_px))
        pos_target = tf.squeeze(tf.gather(target, tp_px))

        neg_pred = tf.squeeze(tf.gather(pred, neg_px))
        neg_target = tf.squeeze(tf.gather(target, neg_px))

        fn_px = tf.compat.v1.where(tf.math.greater(neg_pred, 0.5))
        tn_px = tf.compat.v1.where(tf.math.less_equal(neg_pred, 0.5))

        fn_pred = tf.squeeze(tf.gather(neg_pred, fn_px))
        fn_target = tf.squeeze(tf.gather(neg_target, fn_px))

        tn_pred = tf.squeeze(tf.gather(neg_pred, tn_px))
        tn_target = tf.squeeze(tf.gather(neg_target, tn_px))

        bce = OwnBCE(reduction=tf.keras.losses.Reduction.SUM)

        pos_loss = bce(pos_target, pos_pred)
        fn_loss = bce(fn_target, fn_pred)
        tn_loss = bce(tn_target, tn_pred)

        return ((tf.cast(fn_loss, tf.float32) * self.__weights[0]) + (
                tf.cast(tn_loss, tf.float32) * self.__weights[1])) + \
               (tf.cast(pos_loss, tf.float32) * self.__weights[2])


class OwnBCE(keras.losses.Loss):
    """ Own binary cross entropy loss for the RPN.

    This customs loss function is exactly the same as the keras.losses.binary_crossentropy, but now
    it can be used with predictions and target tensors with one element. Also, instead of returning
    a NaN it returns 0.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__bce = tf.keras.losses.BinaryCrossentropy(*args, **kwargs)

    def call(self, target, pred, *args, **kwargs):
        target = K.switch(tf.math.equal(tf.size(target), 1),
                          tf.expand_dims(target, 0),
                          target)
        pred = K.switch(tf.math.equal(tf.size(pred), 1),
                        tf.expand_dims(pred, 0),
                        pred)

        loss = self.__bce(target, pred)

        loss = K.switch(tf.math.is_nan(loss),
                        tf.constant(0.0),
                        loss)

        return loss


def mrcnn_mask_loss_graph(target_masks, pred, target_class_ids):
    """ Mask binary cross-entropy loss for the masks head.

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
    negative_px = tf.random.shuffle(negative_px)[:tf.size(positive_px)]

    y_true_pos = tf.gather(y_true, positive_px)
    y_pred_pos = tf.gather(y_pred, positive_px)

    y_true_neg = tf.gather(y_true, negative_px)
    y_pred_neg = tf.gather(y_pred, negative_px)

    y_true = tf.concat([y_true_pos, y_true_neg], axis=-1)
    y_pred = tf.concat([y_pred_pos, y_pred_neg], axis=-1)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]

    loss = K.switch(tf.math.greater(tf.size(input=y_true), 0),
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss
