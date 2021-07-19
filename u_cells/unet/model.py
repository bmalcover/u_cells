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

from u_cells.common import config
from u_cells.rpn import model as rpn_model


class ConvBlock(keras_layer.Layer):
    """ Convolutional block used on the encoder

    """

    def __init__(self, layer_idx: int, filters: int, kernel_size: Tuple[int, int],
                 activation: str, kernel_initializer: str = 'he_normal', padding: str = "same",
                 batch_normalization: bool = False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.__layer_idx = layer_idx
        self.__is_batch_normalized = batch_normalization
        self.__kernel_size = kernel_size
        self.__activation = activation

        self.conv2d_1 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)

        if batch_normalization:
            self.batch_normalization_1 = keras_layer.BatchNormalization()

        self.conv2d_2 = keras_layer.Conv2D(filters=filters, kernel_size=kernel_size,
                                           kernel_initializer=kernel_initializer, padding=padding,
                                           activation=activation)
        self.batch_normalization_2 = keras_layer.BatchNormalization()

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

        self.up_sampling_1 = keras_layer.UpSampling2D(size=filter_size)
        self.conv2d_1 = keras_layer.Conv2D(filters, kernel_size=filter_size,
                                           activation=activation, padding=padding,
                                           kernel_initializer=kernel_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.up_sampling_1(x)
        x = self.conv2d_1(x)

        return x


class CropConcatBlock(keras_layer.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                             height_diff: (x2_shape[1] + height_diff),
                             width_diff: (x2_shape[2] + width_diff),
                             :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


class UNet:
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]], out_channel: int,
                 batch_normalization: bool, config_net: config.Config = None, rpn: bool = False,
                 regressor: bool = False, time_distributed: bool = False):

        self.__input_size: Tuple[int, int, int] = input_size
        self.__batch_normalization: bool = batch_normalization
        self.__n_channels: int = out_channel
        self.__config = config_net

        self.__build_rpn: bool = rpn
        self.__build_regressor: bool = regressor
        self.__time_distributed: bool = time_distributed

        self.__internal_model = None
        self.__history = None

    def __build_cells_regressors(self, start_layer, initial_block_id: int, n_filters: int,
                                 dilation_rate: int):
        """ Builds the regressor task.

        This branch of the neural network is an addition to the original U-Net architecture. The
        main goal of this branch is to obtain the number of cells on an image. To do so is based on
        the regression principles, using  special kind of loss function to do it.

        Args:
            start_layer:
            initial_block_id:
            n_filters:
            dilation_rate:

        Returns:

        """
        layers = []
        last_layer = start_layer

        block_id = initial_block_id + 1
        last_layer = keras_layer.Conv2D(n_filters * (2 ** 3), (3, 3), dilation_rate=dilation_rate,
                                        activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal', name=f"conv_{block_id}")(
            last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Conv2D(n_filters * (2 ** 3), (3, 3), dilation_rate=dilation_rate,
                                        activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        name=f"conv_{block_id}_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.MaxPooling2D(pool_size=(2, 2), data_format='channels_last',
                                              name=f"mp_{block_id}")(last_layer)
        layers.append(last_layer)

        block_id = initial_block_id + 2
        last_layer = keras_layer.Conv2D(n_filters * (2 ** 4), (3, 3), dilation_rate=dilation_rate,
                                        activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        name=f"conv_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Conv2D(n_filters * (2 ** 4), (3, 3), dilation_rate=dilation_rate,
                                        activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        name=f"conv_{block_id}_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Flatten()(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Dense(1024, name="dense_1")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Dense(1024, name="dense_2")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Dense(1, name="regressor_output")(last_layer)
        layers.append(last_layer)

        return layers, block_id

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

        if not self.__build_rpn:
            model = keras_model.Model(inputs=input_image, outputs=conv10)
        else:
            if config is None:
                raise AttributeError("Config for RPN model not defined")
            # We connect the U-Net to the RPN via the last CONV5 layer, the last layer of the
            # decoder.
            rpn = rpn_model.build_rpn_model(depth=n_filters * 16)  # Conv5
            rpn_output = rpn([list(encoder.values())[-1][-1]])

            # RPN Output
            rpn_class_logits, rpn_class, rpn_bbox = rpn_output

            # RPN GT
            input_rpn_match = keras_layer.Input(shape=[None, 1], name="input_rpn_match",
                                                dtype=tf.int32)
            input_rpn_bbox = keras_layer.Input(shape=[None, 4], name="input_rpn_bbox",
                                               dtype=tf.float32)
            input_gt_class_ids = keras_layer.Input(shape=[None], name="input_gt_class_ids",
                                                   dtype=tf.int32)

            # RPN Loss
            rpn_class_loss = keras_layer.Lambda(lambda x: rpn_model.class_loss_graph(*x),
                                                name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = keras_layer.Lambda(lambda x: rpn_model.bbox_loss_graph(*x),
                                               name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])

            mask_loss = keras_layer.Lambda(lambda x: rpn_model.mrcnn_mask_loss_graph(*x),
                                           name="img_out_loss")(
                [input_image, input_gt_class_ids, conv10])

            # Input of the model
            inputs = [input_image, input_rpn_match, input_rpn_bbox, input_gt_class_ids]

            # Output of the model
            outputs = [conv10,
                       mask_loss,
                       rpn_class,
                       rpn_bbox,
                       rpn_class_loss,
                       rpn_bbox_loss]

            model = keras_model.Model(inputs=inputs, outputs=outputs, name='r-unet')

        self.__internal_model = model

    def compile(self, loss_func: Union[str, Callable] = "categorical_crossentropy",
                check: bool = False, learning_rate: Union[int, float] = 3e-5, *args, **kwargs):
        """ Compiles the models.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.
            check (bool): Only used in the RPN context. If true checks
            learning_rate

        Returns:

        """
        if not self.__build_rpn:
            loss_functions = {"img_out": loss_func}

            if self.__build_regressor:
                loss_functions['regressor_output'] = 'mean_absolute_error'

            self.__internal_model.compile(*args, **kwargs, optimizer=Adam(lr=learning_rate),
                                          loss=loss_functions, metrics=['categorical_accuracy'])
        else:
            loss_names = ["rpn_class_loss", "rpn_bbox_loss", "img_out_loss"]

            self.__internal_model._losses = []
            self.__internal_model._per_input_losses = {}

            for name in loss_names:
                layer = self.__internal_model.get_layer(name)
                if check and layer.output in self.__internal_model.losses:
                    continue
                loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * 1.0)
                self.__internal_model.add_loss(loss)

            self.__internal_model.compile(*args, **kwargs,
                                          optimizer=Adam(lr=self.__config.LEARNING_RATE),
                                          loss=[loss_func, None, None])

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

        Returns:

        """
        if self.__history is not None:
            warnings.warn("Model already trained, starting new training")

        if self.__build_rpn:
            history = self.__internal_model.fit(train_generator, epochs=epochs,
                                                verbose=verbose,
                                                steps_per_epoch=self.__config.STEPS_PER_EPOCH,
                                                callbacks=callbacks, validation_data=val_generator,
                                                validation_steps=self.__config.VALIDATION_STEPS,
                                                *args, **kwargs)
        else:
            if callbacks is None:
                callbacks = []

            if check_point_path is not None:
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
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

    @property
    def model(self):
        return self.__internal_model

    @property
    def history(self):
        return self.__history

    def predict(self, *args, **kwargs):
        return self.__internal_model.predict(*args, **kwargs)
