# -*- coding: utf-8 -*-
""" Block to build the decoder of the U-Net.

UpConvBlock consists of the combination of an UpSampling2D and a Conv2D. This configuration could
be changes without modifying the results to a Conv2D transposed. Nonetheless and to follow the
original paper, the UpConvBlock is implemented as an UpSampling2D follow up for a Conv2D.

This block doubles the size of the input tensor.

Written by: Miquel Miró Nicolau (UIB)
"""

from typing import Tuple

import tensorflow.keras.layers as keras_layer


class UpConvBlock(keras_layer.Layer):
    """ Block to build the decoder.

    The decoder is build with the combination of UpSampling2D and Conv2D.
    """

    def __init__(self, layer_idx: int, filter_size: Tuple[int, int], filters: int,
                 activation: str = 'relu', padding: str = "same",
                 kernel_initializer: str = "he_normal", **kwargs):
        super(UpConvBlock, self).__init__(**kwargs)

        self.__layer_idx = layer_idx
        self.__filter_size = filter_size
        self.__filters = filters
        self.__activation = activation
        self.__padding: str = padding
        self.__kernel_initializer: str = kernel_initializer

        self.up_sampling_1 = keras_layer.UpSampling2D(size=filter_size)
        self.conv2d_1 = keras_layer.Conv2D(filters, kernel_size=filter_size,
                                           activation=activation, padding=padding,
                                           kernel_initializer=kernel_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.up_sampling_1(x)
        x = self.conv2d_1(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layer_idx': self.__layer_idx,
            'filter_size': self.__filter_size,
            'filters': self.__filters,
            'activation': self.__activation,
            'padding': self.__padding,
            'kernel_initializer': self.__kernel_initializer
        })
        return config
