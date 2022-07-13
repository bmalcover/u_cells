""" Module containing the Merge Mask class.

Merge Mask class is a branch for the U-RPN model. It is used to merge the output of a multichanel
decoder into one channel.

Written by: Miquel Miró Nicolau (UIB), 2022.
"""
from typing import Callable, Union

import tensorflow as tf
from tensorflow.keras import layers as keras_layer


class MergeMasks(tf.keras.Model):
    """Merge Mask class.

    Merge Mask class is a branch for the U-RPN model. It is used to merge the output of a
    multichanel output into a single channel image.

    Attributes:
        filters: Number of filters in the convolutional layer.
        dilation_rate: Dilation rate for the convolutional layer.
        last_activation: Activation function for the convolutional layer.
    """

    def __init__(
        self,
        filters: int,
        dilation_rate: int,
        last_activation: Union[Callable, str],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._dilation_rate = dilation_rate
        self._last_activation = last_activation
        self._filters = filters

    def call(self, inputs) -> keras_layer.Layer:
        merge_branch = keras_layer.Conv2D(
            filters=self._filters,
            kernel_size=(1, 1),
            activation=self._last_activation,
            padding="same",
            dilation_rate=self._dilation_rate,
            kernel_initializer="he_normal",
        )(inputs)

        merge_branch = keras_layer.Lambda(
            lambda t: tf.reduce_sum(t, axis=-1), name="merge_branch"
        )(merge_branch)

        return merge_branch
