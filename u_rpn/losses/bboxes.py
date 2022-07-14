""" Losses for bounding boxes.

The defined loss functions on this module are used to calculate the loss of bounding boxes. In
particular are used for the RPN module from the Faster R-CNN paper.

Wr
"""
import tensorflow as tf
import tensorflow.keras.backend as keras


def positive_cce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Categorical cross entropy only for the slices with some positive element.

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

    loss = keras.switch(
        tf.size(input=y_true) > 0,
        keras.binary_crossentropy(target=y_true, output=y_pred),
        tf.constant(0.0),
    )
    loss = keras.mean(loss)

    return loss


def smooth_l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Implements Smooth-L1 loss.

    y_true and y_pred are typically: [N, 4], but could be any shape.

    Args:
        y_true: Ground truth bounding boxes, e.g. [None, 4] for [x1, y1, x2, y2]
        y_pred: Predicted bounding boxes, e.g. [None, 4] for [x1, y1, x2, y2]

    Returns:
        Smooth-L1 Loss value.
    """
    diff = keras.abs(y_true - y_pred)
    less_than_one = keras.cast(keras.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    return loss


def class_loss_graph(rpn_match: tf.Tensor, rpn_class_logits: tf.Tensor) -> tf.Tensor:
    """RPN anchor classifier loss.

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
    loss = keras.sparse_categorical_crossentropy(
        target=anchor_class, output=rpn_class_logits, from_logits=True
    )
    loss = keras.switch(tf.size(input=loss) > 0, keras.mean(loss), tf.constant(0.0))

    return loss


def bbox_loss_graph(
    target_bbox: tf.Tensor,
    rpn_match: tf.Tensor,
    rpn_bbox: tf.Tensor,
    batch_size: int = 3,
) -> tf.Tensor:
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
        """Picks different number of values from each row in x depending on the values in counts.

        TODO: Make num_rows (batch size) compatible with tf 2.7

        Args:
            x:
            counts:
            num_rows:

        Returns:

        """
        outputs = []
        for i in range(5):
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
