# -*- coding: utf-8 -*-
""" Module containing all functions to build the U-Net model.

This module contains the set of functions that defines the original U-Net networks. This network was
proposed by Ronnenberger et al. and is based on a Encoder-Decoder architecture.
"""

from typing import Callable, Union, Tuple
import warnings

import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layer
from tensorflow.keras.optimizers import *
import tensorflow as tf


class ConvBlock(keras_layer.Layer):
    """ Convolutional block used on the encoder

    """

    def __init__(self, layer_idx: int, filters: int, kernel_size: Tuple[int, int],
                 activation: str, kernel_initializer: str = 'he_normal', padding: str = "same",
                 batch_normalization: bool = False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.__layer_idx: int = layer_idx
        self.__filters: int = filters
        self.__kernel_size = kernel_size
        self.__activation: str = activation
        self.__kernel_initializer: str = kernel_initializer
        self.__padding: str = padding
        self.__is_batch_normalized: bool = batch_normalization

        self.conv2d_1 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)

        if batch_normalization:
            self.batch_normalization_1 = keras_layer.BatchNormalization()

        self.conv2d_2 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)
        self.batch_normalization_2 = keras_layer.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layer_idx': self.__layer_idx,
            'filters': self.__filters,
            'kernel_size': self.__kernel_size,
            'activation': self.__activation,
            'kernel_initializer': self.__kernel_initializer,
            'padding': self.__padding,
            'batch_normalization': self.__is_batch_normalized
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
        self.__padding = padding
        self.__kernel_initializer = kernel_initializer

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
            'kernel_initializer': self.__kernel_initializer,
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


class UNet:
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]], out_channel: int,
                 batch_normalization: bool):

        self.__input_size: Tuple[int, int, int] = input_size
        self.__batch_normalization: bool = batch_normalization
        self.__n_channels: int = out_channel

        self.__internal_model = None
        self.__history = None

    def build_unet(self, n_filters, last_activation: Union[Callable, str], dilation_rate: int = 1,
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
        # Define input batch shape
        input_image = keras_layer.Input(self.__input_size, name="input_image")
        encoder = {}

        conv_params = dict(filters=n_filters,
                           kernel_size=kernel_size,
                           activation='relu',
                           batch_normalization=True)

        x = input_image
        layer_idx = 0

        for layer_idx in range(0, layer_depth):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            if layer_idx == (layer_depth - 1):
                x = ConvBlock(layer_idx, **conv_params, name="embedded_layer")(x)
            else:
                x = ConvBlock(layer_idx, **conv_params)(x)
            encoder[layer_idx] = x
            x = keras_layer.MaxPooling2D(pool_size)(x)

        for layer_idx in range(layer_idx, -1, -1):
            conv_params['filters'] = n_filters * (2 ** layer_idx)

            x = UpConvBlock(layer_idx, filter_size=(2, 2), filters=n_filters * (2 ** layer_idx),
                            activation='relu')(x)
            x = CropConcatBlock()(x, encoder[layer_idx])
            x = ConvBlock(layer_idx, **conv_params)(x)

        conv10 = keras_layer.Conv2D(self.__n_channels, (1, 1), activation=last_activation,
                                    padding='same', dilation_rate=dilation_rate,
                                    kernel_initializer='he_normal', name="img_out")(x)

        model = keras_model.Model(inputs=input_image, outputs=conv10)

        self.__internal_model = model

    def compile(self, loss_func: Union[str, Callable] = "categorical_crossentropy",
                learning_rate: Union[int, float] = 3e-5, *args, **kwargs):
        """ Compiles the models.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.
            learning_rate (Num). Learning rate of the training

        Returns:

        """
        loss_functions = {"img_out": loss_func}

        self.__internal_model.compile(*args, **kwargs, optimizer=Adam(lr=learning_rate),
                                      loss=loss_functions, metrics=['categorical_accuracy'])

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None, verbose=1,
              *args, **kwargs):
        """ Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters. The info is defined
        on Config class or instead passed as parameters.

        Args:
            train_generator:
            val_generator:
            epochs:
            steps_per_epoch:
            validation_steps:
            check_point_path:
            callbacks:
            verbose:

        Returns:

        """
        if self.__history is not None:
            warnings.warn("Model already trained, starting new training")

        if callbacks is None:
            callbacks = []

        if check_point_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                                save_weights_only=False,
                                                                save_best_only=True))

        if val_generator is not None:
            history = self.__internal_model.fit(train_generator, validation_data=val_generator,
                                                epochs=epochs,
                                                validation_steps=validation_steps,
                                                callbacks=callbacks,
                                                steps_per_epoch=steps_per_epoch,
                                                verbose=verbose, *args, **kwargs)
        else:
            history = self.__internal_model.fit(train_generator, epochs=epochs,
                                                callbacks=callbacks, verbose=verbose,
                                                steps_per_epoch=steps_per_epoch, *args,
                                                **kwargs)

        self.__history = history

    def load_weight(self, path: str):
        self.__internal_model.load_weights(path)

    @property
    def model(self):
        return self.__internal_model

    @property
    def history(self):
        return self.__history

    def get_layer(self, *args, **kwargs):
        """ Wrapper of the Keras get_layer function.
        """
        self.__internal_model.get_layer(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """ Infer the value from the Model, wrapper method of the keras predict.

        """
        return self.__internal_model.predict(*args, **kwargs)

    def summary(self):
        self.__internal_model.summary()
