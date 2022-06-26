""" Module containing a set of own build layers

Written by: Miquel Miró Nicolau (UIB)
"""

import tensorflow as tf
import tensorflow.keras.layers as keras_layer

__all__ = ["GradCAM"]


class GradCAM(keras_layer.Layer):
    """GradCAM layer (e.g. gradient between a conv2d and an output)

    This layer calculates how a layer is influenced by a previous one. Its vaguely based on the
    GradCAM implementation by F. Chollet. The big difference is how the gradient is obtained, in
    this case with the special operation of tensorflow gradients.

    Warnings:
         This layer will fail if between the two connected layer there are some layer without
         gradient (a.k.a UpSampling2D).
    """

    @tf.function
    def call(self, conv_layer, output_layer, *args, **kwargs):
        """Calculate the gradient of the output_layer with respect to the conv_layer

        Args:
            conv_layer:
            output_layer:
            *args:
            **kwargs:

        Returns:
            GradCAM layer
        """
        grads = keras_layer.Lambda(
            lambda x: tf.gradients(x[1], x[0], unconnected_gradients="zero")
        )([conv_layer, output_layer])

        # This is a vector where each entry is the mean intensity of the gradient over a specific
        # feature map channel
        pooled_grads = keras_layer.Lambda(lambda x: tf.reduce_mean(x, axis=(1, 2)))(
            grads[0]
        )

        # We multiply each channel in the feature map array by "how important this channel is" with
        # regard to the top predicted class then sum all the channels to obtain the heatmap class
        # activation
        last_conv_layer_output = conv_layer

        pooled_grads = keras_layer.Lambda(
            lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        )(pooled_grads)
        heatmaps = keras_layer.Lambda(lambda x: x[0] * x[1])(
            [last_conv_layer_output, pooled_grads]
        )
        heatmap = keras_layer.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(heatmaps)
        heatmap = keras_layer.Lambda(lambda x: tf.expand_dims(x, axis=-1))(heatmap)

        return heatmap
