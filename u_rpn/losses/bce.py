""" Segmentation losses based on binary cross-entropy (BCE) for the training of FCN models.

The losses defined in this module has as the main feature the ability to be used for the comparasion
between a ground truth and a predicted segmentation. All the data is supposed to be a tensor of at
least three dimensions. The first dimension is the height, the second is the width and the third is
is the channel.

All losses of this module are modification on the default BCE loss function.

Losses:
    WeightedBCE: Weighted positive and negative BCE loss adapted to the U-RPN architecture.
    WeightedTernaryBCE: Weighted positive, true negative and false negative BCE loss adapted to the
                        U-RPN architecture.
    WeightedQuaternaryBCE: Weighted true positive, false positive, true negative and false negative
                            BCE loss adapted to the U-RPN architecture.

Writen by: Miquel Miró Nicolau (UIB)
"""

from typing import List, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras


def conditional_dim_increment(tensor: tf.Tensor) -> tf.Tensor:
    """Conditional dimension increment

    Args:
        tensor (tf.Tensor):

    Returns:
        Tensor conditional dimension increment.
    """
    return K.switch(
        tf.math.equal(tf.size(tensor), 1), tf.expand_dims(tensor, 0), tensor
    )


class OwnBCE(keras.losses.Loss):
    """Own binary cross entropy loss for the RPN.

    This customs loss function is exactly the same as the keras.losses.binary_crossentropy, but now
    it can be used with predictions and target tensors with one element. Also, instead of returning
    a NaN it returns 0.
    """

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)

        self.__bce = tf.keras.losses.BinaryCrossentropy(*args, **kwargs)

    def call(
        self, target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict
    ) -> tf.Tensor:
        target = conditional_dim_increment(target)
        pred = conditional_dim_increment(pred)

        loss = self.__bce(target, pred)

        loss = K.switch(tf.math.is_nan(loss), tf.constant(0.0), loss)

        return loss


class WeightedBCE(keras.losses.Loss):
    """Weighted binary cross entropy loss for the RPN.

    To fix unbalanced classes, we split the loss function into positive and negatives classes, and
    we weight the loss of each class with the number of pixels within.
    """

    def __init__(self, positive_weight: float = 0.5):
        super().__init__()
        self.__positive_w = positive_weight

    def call(
        self, target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict
    ) -> tf.Tensor:
        """Weighted binary cross entropy loss for the RPN.

        To fix unbalanced classes, we split the loss function into positive and negatives classes,
        and we weight the loss of each class with the number of pixels within.

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
            tf.cast(pos_loss, tf.float32) * (1 - self.__positive_w)
        )


class WeightedTernaryBCE(keras.losses.Loss):
    """Weighted ternary binary cross entropy loss for the RPN.

    To fix unbalanced classes, we split the loss function into positives, false negative,
    true negatives classes, and we weight the loss of each class with the number
    of pixels within.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__()

        if weights is None:
            weights = [0.3, 0.3, 0.3]
        self.__weights = weights

    def call(
        self, target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict
    ) -> tf.Tensor:
        """Ternary weighted binary cross entropy loss for the RPN.

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

        return (
            (tf.cast(fn_loss, tf.float32) * self.__weights[0])
            + (tf.cast(tn_loss, tf.float32) * self.__weights[1])
        ) + (tf.cast(pos_loss, tf.float32) * self.__weights[2])


class WeightedTernaryBCEReverse(keras.losses.Loss):
    """Weighted reverse ternary binary cross entropy loss for the RPN.

    To fix unbalanced classes, we split the loss function into negatives, false positive,
    true positives classes, and we weight the loss of each class with the number
    of pixels within.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__()

        if weights is None:
            weights = [0.3, 0.3, 0.3]
        self.__weights: List[float] = weights

    def call(
        self, target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict
    ) -> tf.Tensor:
        """Ternary weighted binary cross entropy loss for the RPN.

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

        fp_px = tf.compat.v1.where(tf.math.greater(pos_pred, 0.5))
        tp_px = tf.compat.v1.where(tf.math.less_equal(pos_pred, 0.5))

        fp_pred = tf.squeeze(tf.gather(pos_pred, fp_px))
        fp_target = tf.squeeze(tf.gather(pos_target, fp_px))

        tp_pred = tf.squeeze(tf.gather(pos_pred, tp_px))
        tp_target = tf.squeeze(tf.gather(pos_target, tp_px))

        bce = OwnBCE(reduction=tf.keras.losses.Reduction.SUM)

        neg_loss = bce(neg_target, neg_pred)
        fp_loss = bce(fp_target, fp_pred)
        tp_loss = bce(tp_target, tp_pred)

        return (
            (tf.cast(neg_loss, tf.float32) * self.__weights[0])
            + (tf.cast(fp_loss, tf.float32) * self.__weights[1])
        ) + (tf.cast(tp_loss, tf.float32) * self.__weights[2])


class WeightedQuaternaryBCE(keras.losses.Loss):
    """Weighted quaternary binary cross entropy loss for the RPN.

    To fix unbalanced classes, we split the loss function into false positives, true positive,
    false negative, true negatives classes, and we weight the loss of each class with the number
    of pixels within.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__()

        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]
        self.__weights = weights

    def call(
        self, target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict
    ) -> tf.Tensor:
        """Quaternary weighted binary cross entropy loss for the RPN.

        To fix unbalanced classes, we split the loss function into false positives, true positive,
        false negative, true negatives classes, and we weight the loss of each class with the number
        of pixels within.

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

        pos_target = conditional_dim_increment(pos_target)
        pos_pred = conditional_dim_increment(pos_pred)

        neg_pred = tf.squeeze(tf.gather(pred, neg_px))
        neg_target = tf.squeeze(tf.gather(target, neg_px))

        neg_target = conditional_dim_increment(neg_target)
        neg_pred = conditional_dim_increment(neg_pred)

        fn_px = tf.compat.v1.where(tf.math.greater(neg_pred, 0.5))
        tn_px = tf.compat.v1.where(tf.math.less_equal(neg_pred, 0.5))

        fn_pred = tf.squeeze(tf.gather(neg_pred, fn_px))
        fn_target = tf.squeeze(tf.gather(neg_target, fn_px))

        tn_pred = tf.squeeze(tf.gather(neg_pred, tn_px))
        tn_target = tf.squeeze(tf.gather(neg_target, tn_px))

        fp_px = tf.compat.v1.where(tf.math.greater(pos_pred, 0.5))
        tp_px = tf.compat.v1.where(tf.math.less_equal(pos_pred, 0.5))

        fp_pred = tf.squeeze(tf.gather(pos_pred, fp_px))
        fp_target = tf.squeeze(tf.gather(pos_target, fp_px))

        tp_pred = tf.squeeze(tf.gather(pos_pred, tp_px))
        tp_target = tf.squeeze(tf.gather(pos_target, tp_px))

        bce = OwnBCE(reduction=tf.keras.losses.Reduction.SUM)

        fp_loss = bce(fp_target, fp_pred)
        tp_loss = bce(tp_target, tp_pred)
        fn_loss = bce(fn_target, fn_pred)
        tn_loss = bce(tn_target, tn_pred)

        loss = (
            tf.cast(fp_loss, tf.float64) * self.__weights[0]
            + tf.cast(tp_loss, tf.float64) * self.__weights[1]
            + tf.cast(fn_loss, tf.float64) * self.__weights[2]
            + tf.cast(tn_loss, tf.float64) * self.__weights[3]
        )

        return loss


class WU4BCE(WeightedQuaternaryBCE):
    """Weighted Unsorted quaternary binary cross-entropy

    To fix unbalanced classes, we split the loss function into false positives, true positive,
    false negative, true negatives classes, and we weight the loss of each class with the number
    of pixels within. This loss function is able to handle unsorted classes.

    """

    def __qbce_by_channel(self, targets, pred_tensor) -> tf.Tensor:
        """Compute the quaternary binary cross-entropy loss for each channel.

        Args:
            targets: Tensor of shape [num_channels, height, width]
            pred_tensor: Tensor of shape [height, width]

        Returns:
            Tensor of shape [num_channels]
        """
        fn_call = super().call

        loss = tf.map_fn(fn=lambda x: fn_call(x, pred_tensor), elems=targets)

        loss_norm = tf.nn.softmax(tf.reduce_max(loss) - loss)

        return tf.reduce_sum(loss * loss_norm, axis=-1)

    @tf.autograph.experimental.do_not_convert
    def call(
        self, target: tf.Tensor, pred: tf.Tensor, *args: list, **kwargs: dict
    ) -> tf.Tensor:
        original_shape = tf.shape(target)
        target = tf.cast(target, tf.float64)
        pred = tf.cast(pred, tf.float64)

        target = tf.transpose(target, (0, 3, 1, 2))
        pred = tf.transpose(pred, (0, 3, 1, 2))

        target = tf.reshape(target, (-1, original_shape[1], original_shape[2]))
        pred = tf.reshape(pred, (-1, original_shape[1], original_shape[2]))

        loss_by_channel = tf.map_fn(
            fn=lambda x: self.__qbce_by_channel(target, x), elems=pred
        )

        return tf.reduce_mean(loss_by_channel)
