""" Module containing focal loss function.

Writen by: Miquel Miró Nicolau (UIB)
"""
import tensorflow as tf
import tensorflow.keras.backend as keras
import tensorflow_addons as tfa

from ..losses import segmentation as seg_losses

fl = tfa.losses.SigmoidFocalCrossEntropy(gamma=2)


def own_focal_loss(target_masks, target_class_ids, pred_masks):
    """Combines the sigmoid focal loss and the dice coefficient loss

    Args:
        target_masks: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1.
        target_class_ids: By convention
        pred_masks: ([batch, num_rois, height, width])  A float32 tensor of values 0 or 1.

    Returns:

    """
    loss = 10.0 * fl(target_masks, pred_masks) - tf.math.log(
        seg_losses.dice_expanded(target_masks, pred_masks)
    )
    loss = keras.mean(loss)

    return loss
