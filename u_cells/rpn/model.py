import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow as tf


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before
                          softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride=1, anchors_per_location=3, depth=256):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.
    
    Args:
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.
    
    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


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
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.compat.v1.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def bbox_loss_graph(target_bbox, rpn_match, rpn_bbox):
    """

    Args:
        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))]. Uses 0 padding to
                    fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=pos, -1=neg, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:

    """

    def batch_pack_graph(x, counts, num_rows):
        """ Picks different number of values from each row in x depending on the values in counts.

        Args:
            x:
            counts:
            num_rows:

        Returns:

        """
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.compat.v1.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    # IMAGE_PER_GPU = 4 , from the sample
    target_bbox = batch_pack_graph(target_bbox, batch_counts, 4)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def smooth_l1_loss(y_true, y_pred):
    """ Implements Smooth-L1 loss.

    y_true and y_pred are typically: [N, 4], but could be any shape.

    Args:
        y_true:
        y_pred:

    Returns:

    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    return loss
