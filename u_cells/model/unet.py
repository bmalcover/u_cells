# -*- coding: utf-8 -*-
""" Module containing all functions to build the U-Net model.

This module contains the set of functions that defines the original U-Net networks. This network was
proposed by Ronnenberger et al. and is based from an Encoder-Decoder architecture.

Written by: Miquel Miró Nicolau (UIB)
"""
from typing import Callable, Union, Tuple

import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layer
from tensorflow.keras.optimizers import *
import tensorflow as tf

from ..model.base_model import BaseModel
from .. import layers as own_layer


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
            x = self.batch_normalization_1(x)

        x = self.conv2d_2(x)

        if self.__is_batch_normalized:
            x = self.batch_normalization_2(x)

        if self.__residual:
            shortcut = self.shortcut(inputs)

            if self.__is_batch_normalized:
                shortcut = self.shortcut_bn(shortcut)
            x = keras_layer.Add()([x, shortcut])

        return x

    @property
    def layer_idx(self):
        return self.__layer_idx


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


class CropConcatBlock(keras_layer.Layer):

    @tf.autograph.experimental.do_not_convert
    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:, height_diff: (x2_shape[1] + height_diff),
                             width_diff: (x2_shape[2] + width_diff), :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


class UNet(BaseModel):
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]], out_channel: int,
                 batch_normalization: bool, residual: bool = False):
        super().__init__(input_size)

        self.__batch_normalization: bool = batch_normalization
        self.__n_channels: int = out_channel
        self.__residual = residual

    def build(self, n_filters, last_activation: Union[Callable, str], dilation_rate: int = 1,
              layer_depth: int = 5, kernel_size: Tuple[int, int] = (3, 3),
              pool_size: Tuple[int, int] = (2, 2)):
        """ Builds the graph and model for the U-Net.

        The U-Net, first introduced by Ronnenberger et al., is an encoder-decoder architecture.
        Build through the stack of 2D convolutional and up sampling 2D.

        Args:
            n_filters:
            last_activation:
            dilation_rate:
            layer_depth:
            kernel_size:
            pool_size:

        """
        self._layers = {}

        encoder = EncoderUNet(input_size=self._input_size, residual=self.__residual)
        input_image, embedded = encoder.build(n_filters=n_filters, last_activation=last_activation,
                                              pool_size=pool_size, dilation_rate=dilation_rate,
                                              layer_depth=layer_depth, kernel_size=kernel_size)

        self._layers['encoder'] = encoder

        decoder = DecoderUNet(input_size=self._input_size, residual=self.__residual,
                              n_channels=self.__n_channels)
        mask_out = decoder.build(n_filters=n_filters, last_activation=last_activation,
                                 encoder=encoder, dilation_rate=dilation_rate,
                                 kernel_size=kernel_size, embedded=embedded)

        self._layers['decoder'] = decoder

        model = keras_model.Model(inputs=input_image, outputs=mask_out)

        self._internal_model = model

        return input_image, encoder, decoder

    def compile(self, loss_func: Union[str, Callable] = "categorical_crossentropy",
                learning_rate: Union[int, float] = 3e-5, *args, **kwargs):
        """ Compiles the model.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.
            learning_rate (Num). Learning rate of the training

        Returns:

        """
        loss_functions = {"img_out": loss_func}

        self._internal_model.compile(*args, **kwargs, optimizer=Adam(lr=learning_rate),
                                     loss=loss_functions, metrics=['categorical_accuracy'])


class EncoderUNet(BaseModel):
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 residual: bool = False):
        super().__init__(input_size)

        self.__residual = residual
        self._layers = None

    def build(self, n_filters, last_activation: Union[Callable, str], dilation_rate: int = 1,
              layer_depth: int = 5, kernel_size: Tuple[int, int] = (3, 3),
              pool_size: Tuple[int, int] = (2, 2)):
        # Define input batch shape
        input_image = keras_layer.Input(self._input_size, name="input_image")
        self._layers = {}

        conv_params = dict(filters=n_filters,
                           kernel_size=kernel_size,
                           activation='relu',
                           residual=self.__residual,
                           batch_normalization=True)

        x = input_image

        for layer_idx in range(0, layer_depth):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            x = ConvBlock(layer_idx, name=f"e_conv_block_{layer_idx}", **conv_params)(x)
            self._layers[layer_idx] = x

            x = keras_layer.MaxPooling2D(pool_size, name=f"e_max_pool_{layer_idx}")(x)

        return input_image, x


class DecoderUNet(BaseModel):
    """ Decoder of the U-Net.

    This class is responsible for the decoder part of the U-Net. It is a wrapper for the keras.model
    In addition to the vanilla decoder it also includes the posibility to use more advanced
    architectures with the parameters options:

    Args:
        input_size: Size of the input image.
        residual: Whether to use residual connections in the decoder.
        n_channels: Number of channels of the output image.
        class_output_size: By default None, if not None the decoder will have an extra branch for
            the classification of the output channels between if there are object in the channel or
            only background.
        merge_branch: Boolean, if true adds an extra output to the decoder that merges the masks.
    """

    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]],
                 n_channels: int = 1, residual: bool = False, class_output_size=None,
                 merge_branch: bool = False):
        super().__init__(input_size)

        self.__residual = residual
        self.__n_channels = n_channels
        self.__class_output = class_output_size
        self.__merge_branch = merge_branch

    def build(self, n_filters, last_activation: Union[Callable, str], encoder: EncoderUNet,
              embedded, extra_layer: dict = None, dilation_rate: int = 1,
              kernel_size: Tuple[int, int] = (3, 3), coord_conv=None):
        conv_params = dict(filters=n_filters,
                           kernel_size=kernel_size,
                           activation='relu',
                           residual=self.__residual,
                           batch_normalization=True)

        self._layers = {}
        x = embedded
        for layer_idx in range(len(encoder) - 1, -1, -1):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            x = UpConvBlock(layer_idx, filter_size=(2, 2), filters=n_filters * (2 ** layer_idx),
                            activation='relu', name=f"d_up_conv_block{layer_idx}")(x)

            encoder_layer = encoder[layer_idx]
            if extra_layer is not None and layer_idx in extra_layer:
                encoder_layer = keras_layer.Concatenate(axis=-1,
                                                        name=f"d_concatenate_extra_{layer_idx}")(
                    [encoder_layer, extra_layer[layer_idx]])

            x = keras_layer.Concatenate(axis=-1, name=f"d_concatenate_{layer_idx}")(
                [x, encoder_layer])

            coord_conv_size = None
            if coord_conv is not None and layer_idx in coord_conv:
                coord_conv_size = coord_conv[layer_idx]

            x = ConvBlock(layer_idx, coord_conv=coord_conv_size, name=f"d_conv_block_{layer_idx}",
                          **conv_params)(x)

            self._layers[layer_idx] = x

        out_mask = keras_layer.Conv2D(self.__n_channels, (1, 1), activation=last_activation,
                                      padding='same', dilation_rate=dilation_rate,
                                      kernel_initializer='he_normal', name="img_out")(x)
        if self.__class_output is not None:
            class_out = keras_layer.GlobalAvgPool2D()(x)
            class_out = keras_layer.Dense(self.__class_output, activation='relu',
                                          name='mask_class_1')(class_out)
            class_out = keras_layer.Dropout(0.5)(class_out)
            class_out = keras_layer.Dense(self.__class_output, activation='relu',
                                          name='mask_class_2')(class_out)
            class_out = keras_layer.Dense(self.__n_channels, activation='sigmoid',
                                          name='mask_class_logits_1')(class_out)

            out_mask = keras_layer.Multiply(name="multiply_end")([out_mask, class_out])

            output = [out_mask, class_out]
        else:
            output = [out_mask]

        if self.__merge_branch:
            merge_branch = keras_layer.Lambda(lambda t: tf.reduce_sum(t, axis=-1),
                                              name="merge_branch")(out_mask)
            output.append(merge_branch)

        return tuple(output)
