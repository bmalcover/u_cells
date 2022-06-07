# -*- coding: utf-8 -*-
""" Block to build the encoder of the U-Net.

Build through the combination of Convolutional and MaxPooling layers, the main goal of this block
is to obtain meaningful features from the input data.

The block is always composed of two convolutional layers and one max pooling layer. Then with the
selected flags can also be added batch normalization, coord_conv and residual connections.

Written by: Miquel Miró Nicolau (UIB)
"""

from typing import Tuple

import tensorflow.keras.layers as keras_layer

from .. import layers as own_layer

__all__ = ["ConvBlock"]


class ConvBlock(keras_layer.Layer):
    """ Convolutional block used on the encoder

    """

    def __init__(self, layer_idx: int, filters: int, kernel_size: Tuple[int, int], activation: str,
                 kernel_initializer: str = 'he_normal', padding: str = "same",
                 coord_conv=None, residual: bool = False, batch_normalization: bool = False,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.__layer_idx: int = layer_idx
        self.__filters: int = filters
        self.__kernel_size = kernel_size
        self.__activation: str = activation
        self.__kernel_initializer: str = kernel_initializer
        self.__padding: str = padding
        self.__is_batch_normalized: bool = batch_normalization
        self.__residual = residual
        self.__coord_conv = coord_conv

        self.conv2d_1 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)

        if batch_normalization:
            self.batch_normalization_1 = keras_layer.BatchNormalization()

        if coord_conv is not None:
            self.conv2d_2 = own_layer.CoordConv(x_dim=coord_conv[0], y_dim=coord_conv[1],
                                                filters=filters, kernel_size=kernel_size,
                                                kernel_initializer=kernel_initializer,
                                                padding=padding,
                                                activation=activation)
        else:
            self.conv2d_2 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                               kernel_initializer=kernel_initializer,
                                               padding=padding,
                                               activation=activation)
        self.batch_normalization_2 = keras_layer.BatchNormalization()

        self.shortcut = keras_layer.Conv2D(filters, (1, 1), padding=padding)
        self.shortcut_bn = keras_layer.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layer_idx': self.__layer_idx,
            'filters': self.__filters,
            'kernel_size': self.__kernel_size,
            'activation': self.__activation,
            'kernel_initializer': self.__kernel_initializer,
            'padding': self.__padding,
            'batch_normalization': self.__is_batch_normalized,
            'residual': self.__residual,
            'coord_conv': self.__coord_conv
        })
        return config

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        if self.__is_batch_normalized:
            if training is not None:
                x = self.batch_normalization_1(x, training=training)
            else:
                x = self.batch_normalization_1(x)

        x = self.conv2d_2(x)

        if self.__is_batch_normalized:
            if training is not None:
                x = self.batch_normalization_2(x, training=training)
            else:
                x = self.batch_normalization_2(x)

        if self.__residual:
            shortcut = self.shortcut(inputs)

            if self.__is_batch_normalized:
                if training is not None:
                    shortcut = self.shortcut_bn(shortcut, training=training)
                else:
                    shortcut = self.shortcut_bn(shortcut)
            x = keras_layer.Add()([x, shortcut])

        return x

    @property
    def layer_idx(self):
        return self.__layer_idx

    @property
    def training(self):
        return self
