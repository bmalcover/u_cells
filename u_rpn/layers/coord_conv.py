# -*- coding: utf-8 -*-
""" CoordConv layer, first introduced by Liu et al. (2018)

Refs:
    https://arxiv.org/pdf/1807.03247.pdf%7C

Copyright (c) 2019 Uber Technologies, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adaptation of the original code to the framework of the U-Net by Miquel Miró (UIB)
"""

import tensorflow as tf
from tensorflow.keras import layers as keras_layer

__all__ = ["CoordConv"]


class AddCoords(keras_layer.Layer):
    """Add coords to a tensor"""

    def __init__(self, x_dim=64, y_dim=64, with_r=False, skip_tile=False, *args, **kwargs):
        super(AddCoords, self).__init__(*args, **kwargs)

        self.__x_dim = x_dim
        self.__y_dim = y_dim
        self.__with_r = with_r
        self.__skip_tile = skip_tile

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'x_dim': self.__x_dim,
            'y_dim': self.__y_dim,
            'with_r': self.__with_r,
            'skip_tile': self.__skip_tile
        })

    def call(self, inputs, **kwargs):
        """

        Args:
            inputs: (batch, 1, 1, c), or (batch, x_dim, y_dim, c). In the first case, first tile the
             input_tensor to be (batch, x_dim, y_dim, c). In the second case, skiptile, just concat
            **kwargs:

        Returns:

        """
        if not self.__skip_tile:
            inputs = tf.tile(inputs,
                             [1, self.__x_dim, self.__y_dim, 1])  # (batch, 64, 64, 2)
            inputs = tf.cast(inputs, 'float32')

        batch_size_tensor = tf.shape(inputs)[0]  # get batch size

        xx_ones = tf.ones([batch_size_tensor, self.__x_dim],
                          dtype=tf.int32)  # e.g. (batch, 64)
        xx_ones = tf.expand_dims(xx_ones, -1)  # e.g. (batch, 64, 1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.__y_dim), 0),
                           [batch_size_tensor, 1])  # e.g. (batch, 64)
        xx_range = tf.expand_dims(xx_range, 1)  # e.g. (batch, 1, 64)

        xx_channel = tf.matmul(xx_ones, xx_range)  # e.g. (batch, 64, 64)
        xx_channel = tf.expand_dims(xx_channel, -1)  # e.g. (batch, 64, 64, 1)

        yy_ones = tf.ones([batch_size_tensor, self.__y_dim],
                          dtype=tf.int32)  # e.g. (batch, 64)
        yy_ones = tf.expand_dims(yy_ones, 1)  # e.g. (batch, 1, 64)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.__x_dim), 0),
                           [batch_size_tensor, 1])  # (batch, 64)
        yy_range = tf.expand_dims(yy_range, -1)  # e.g. (batch, 64, 1)

        yy_channel = tf.matmul(yy_range, yy_ones)  # e.g. (batch, 64, 64)
        yy_channel = tf.expand_dims(yy_channel, -1)  # e.g. (batch, 64, 64, 1)

        xx_channel = tf.cast(xx_channel, 'float32') / (self.__x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.__y_dim - 1)
        xx_channel = xx_channel * 2 - 1  # [-1,1]
        yy_channel = yy_channel * 2 - 1

        ret = tf.concat([inputs,
                         xx_channel,
                         yy_channel], axis=-1)  # e.g. (batch, 64, 64, c+2)

        if self.__with_r:
            rr = tf.sqrt(tf.square(xx_channel)
                         + tf.square(yy_channel)
                         )
            ret = tf.concat([ret, rr], axis=-1)  # e.g. (batch, 64, 64, c+3)

        return ret


class CoordConv(keras_layer.Layer):
    """CoordConv layer as in the paper."""

    def __init__(self, x_dim, y_dim, filters, kernel_size, kernel_initializer, padding, activation,
                 *args, **kwargs):
        super(CoordConv, self).__init__(*args, **kwargs)

        self.__x_dim = x_dim
        self.__y_dim = y_dim
        self.__filters: int = filters
        self.__kernel_size = kernel_size
        self.__activation: str = activation
        self.__kernel_initializer: str = kernel_initializer
        self.__padding: str = padding

        self.add_coords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=False, skip_tile=True)
        self.conv = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                       kernel_initializer=kernel_initializer, padding=padding,
                                       activation=activation, *args, **kwargs)

    def call(self, inputs, **kwargs):
        ret = self.add_coords(inputs)
        ret = self.conv(ret)

        return ret

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'x_dim': self.__x_dim,
            'y_dim': self.__y_dim,
            'filters': self.__filters,
            'kernel_size': self.__kernel_size,
            'kernel_initializer': self.__kernel_initializer,
            'padding': self.__padding,
            'activation': self.__activation,
        })
