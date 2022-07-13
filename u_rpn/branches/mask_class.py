import tensorflow as tf
from tensorflow.keras import layers as keras_layer


class MaskClass(tf.keras.Model):
    def __init__(self, filters: int, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filters = filters
        self._channels = channels

    def call(self, input_tensor, training=False):
        """

        Args:
            input_tensor: Tensor of convolutional filters.
            training: Boolean, indicating whether if the layer must be in training mode.

        Returns:

        """
        class_out = keras_layer.GlobalAvgPool2D()(input_tensor)
        class_out = keras_layer.Dense(
            self._filters, activation="relu", name="mask_class_1"
        )(class_out)

        class_out = keras_layer.Dropout(0.5)(class_out, training=training)
        class_out = keras_layer.Dense(
            self._filters, activation="relu", name="mask_class_2"
        )(class_out)

        logits = keras_layer.Dense(
            self._channels, activation="sigmoid", name="mask_class_logits_1"
        )(class_out)

        return logits
