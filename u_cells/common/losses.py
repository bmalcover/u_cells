from typing import Callable, Union

import numpy as np

import tensorflow.keras.backend as K
import tensorflow as tf


def onw_dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (
            K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def own_dice_coef_loss(y_true, y_pred):
    return 1 - onw_dice_coef(y_true, y_pred)


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
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights  # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)

        iou = 1 - numerator / denominator

        return iou

    return loss


def dice_coef_loss(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5):
    """ Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
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
    dice = tf.reduce_mean(dice, name='dice_coe')

    return 1 - dice


def positive_cce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Categorical cross entropy only for the slices with some positive element.

    To calculate the loss only takes into consideration the channels with some value different to 0
    in the ground truth.
    Args:
        y_true (tf.Tensor):
        y_pred (tf.Tensor):

    Returns:

    """
    zero_values = tf.reduce_sum(y_true, -1)
    zero_values = tf.reduce_sum(zero_values, -1)

    y_true = K.reshape(y_true, (-1, y_true.shape[-2], y_true.shape[-1]))
    y_pred = K.reshape(y_pred, (-1, y_true.shape[-2], y_true.shape[-1]))

    positives = K.reshape(zero_values, (-1, 1))[:, 0]
    postives_ix = tf.compat.v1.where(positives > 0)[:, 0]

    y_true = tf.gather(y_true, postives_ix)
    y_pred = tf.gather(y_pred, postives_ix)

    loss = K.switch(tf.size(input=y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)

    return loss
