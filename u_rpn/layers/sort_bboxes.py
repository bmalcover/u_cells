""" Layer to sort bboxes according to their position.

Written by: Miquel Miró Nicolau (UIB), 2022.
"""
import tensorflow as tf
import tensorflow.keras.layers as keras_layer

__all__ = ["SortBboxes"]


class SortBboxes(keras_layer.Layer):
    def __init__(
        self,
        tolerance_factor: int = 100,
        height: int = 100,
        order: str = "ASCENDING",
        *args,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.__tolerance_factor = tolerance_factor
        self.__height = height
        self.__order = order

    def get_config(self) -> dict:
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstantiated later (without its trained weights) from this
        configuration.

        The config of a layer does not include connectivity information, nor the layer class name.
        These are handled by `Network` (one layer of abstraction above).

        Note that `get_config()` does not guarantee to return a fresh copy of dict every time it is
        called. The callers should make a copy of the returned dict if they want to modify it.

        Returns:
            Python dictionary.
        """
        config = super().get_config().copy()
        config.update(
            {
                "tolerance_factor": self.__tolerance_factor,
                "height": self.__height,
                "order": self.__order,
            }
        )

        return config

    def get_precedence(self, tensor: tf.Tensor) -> tf.Tensor:
        """From a set of bboxes obtains a single number to sort them.

        Args:
            tensor:

        Returns:

        """
        tolerance_factor = self.__tolerance_factor

        return (
            (tensor[:, :, 0] // tolerance_factor) * tolerance_factor
        ) * self.__height + tensor[:, :, 1]

    def call(self, inputs: tf.Tensor, *args: list, **kwargs: dict) -> tf.Tensor:
        """Feed forward method.

        From a set of bounding boxes obtains a number to sort them.

        Args:
            inputs: Tensor of shape [batch_size, number of bboxes, 4]
            *args:
            **kwargs:

        Returns:
            Ordered tensor
        """
        key_s = self.get_precedence(inputs)
        key_ss = tf.argsort(key_s, direction=self.__order)

        inputs = tf.gather(inputs, key_ss, batch_dims=1, axis=1)
        return inputs
