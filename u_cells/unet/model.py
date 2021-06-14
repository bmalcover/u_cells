# -*- coding: utf-8 -*-
""" Module containing all functions to build the U-Net model.

This module contains the set of functions that defines the original U-Net networks. This network was proposed by
Ronnenberger et al. and is based on a Encoder-Decoder architecture.
"""

from typing import Callable, Union, Tuple, List
import warnings

from tensorflow.keras.optimizers import *
import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layer
import tensorflow as tf

from u_cells.common import config
from u_cells.rpn import model as rpn_model


class UNet:
    def __init__(self, input_size: Union[Tuple[int, int, int], Tuple[int, int]], out_channel: int,
                 batch_normalization: bool, config_net: config.Config = None, rpn: bool = False,
                 regressor: bool = False):

        self.__input_size: Tuple[int, int, int] = input_size
        self.__batch_normalization: bool = batch_normalization
        self.__build_rpn: bool = rpn
        self.__build_regressor: bool = regressor
        self.__n_channels: int = out_channel
        self.__config = config_net

        self.__internal_model = None
        self.__is_trained = False

    def __build_encoder(self, n_filters: int, start_layer, n_blocks: int, name: str = "encode",
                        dilation_rate: int = 1, initial_block_id: int = 0):
        """ Build encoder for U-Net.

        The encoder has the form of a traditional CNN. The start layer must be an Input image while
        the last layer output will an embedded vector coding the information of the image.

        Args:
            n_filters:
            start_layer:
            n_blocks:
            name:
            dilation_rate:
            initial_block_id:

        Returns:
            encoder (dict): Dictionary containing the layers that defines the model. Each key-value is a different
                            block of the encoder. Each value is a list containing all the layers (in order) of that
                            block.
            block_id (int): Last id used to define a name of a layer.
        """
        bn: bool = self.__batch_normalization

        layers = {}
        prev_layer = start_layer
        block_id = initial_block_id
        for i in range(initial_block_id, initial_block_id + n_blocks):
            block_id = i + 1
            dict_key = name + "_" + str(block_id)
            layers[dict_key] = []

            n_filters_layer = n_filters * (2 ** i)

            conv1 = keras_layer.Conv2D(n_filters_layer, (3, 3), activation='relu', padding='same',
                                       dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                       name=f"conv_{block_id}")(prev_layer)
            layers[dict_key].append(conv1)

            if bn:
                conv1 = keras_layer.BatchNormalization(name=f"bn_conv_{block_id}")(conv1)
                layers[dict_key].append(conv1)

            conv1 = keras_layer.Conv2D(n_filters_layer, (3, 3), activation='relu', padding='same',
                                       dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                       name=f"conv_{block_id}_{block_id}")(conv1)
            layers[dict_key].append(conv1)

            if bn:
                conv1 = keras_layer.BatchNormalization(name=f"bn_conv_{block_id}_{block_id}")(conv1)
                layers[dict_key].append(conv1)

            if (i + 1) != initial_block_id + n_blocks:
                pool1 = keras_layer.MaxPooling2D(pool_size=(2, 2), data_format='channels_last',
                                                 name=f"mp_{block_id}")(conv1)
                layers[dict_key].append(pool1)

                prev_layer = pool1

        return layers, block_id

    def __build_decoder(self, encoder, filters: List[int], name: str = "decode",
                        dilation_rate: int = 1, initial_block_id: int = 0):
        """ Build the decoder model.

        The decoder model of the U-Net is build upon the conjunction of UpSampling layers with Conv2D layers. The first
        ones increase the size of the input using some "shallow" technique. After applying it the Conv2D refines the
        output. The input of each block is the result of the previous block concatenate it with a feature map of the
        encoder.

        Args:
            encoder:
            filters:
            name:
            dilation_rate:
            initial_block_id:

        Returns:

        """
        bn: bool = self.__batch_normalization

        layers = {}
        prev_layer = list(encoder.values())[-1][-1]

        encoder_layers = list(encoder.values())[:-1]
        encoder_layers = encoder_layers[::-1]
        for i, (filter_per_layer, enc_layer) in enumerate(zip(filters, encoder_layers)):
            block_id: int = initial_block_id + i
            dict_key = name + "_" + str(block_id)

            layers[dict_key] = []

            up6 = keras_layer.concatenate(
                [keras_layer.Conv2D(filter_per_layer, (2, 2), activation='relu', padding='same',
                                    dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                    name=f"conv_{block_id}")(
                    keras_layer.UpSampling2D(size=(2, 2), name=f"up_{block_id}")(prev_layer)),
                    enc_layer[-2]], name=f"conct_{block_id}", axis=3)
            layers[dict_key].append(up6)

            conv6 = keras_layer.Conv2D(filter_per_layer, (3, 3), activation='relu', padding='same',
                                       dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                       name=f"conv_{block_id}_{block_id}")(up6)
            layers[dict_key].append(conv6)

            if bn:
                conv6 = keras_layer.BatchNormalization(name=f"bn_conv_{block_id}")(conv6)
                layers[dict_key].append(conv6)

            conv6 = keras_layer.Conv2D(filter_per_layer, (3, 3), activation='relu', padding='same',
                                       dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                       name=f"conv_{block_id}_{block_id}_{block_id}")(conv6)
            layers[dict_key].append(conv6)

            if bn:
                conv6 = keras_layer.BatchNormalization(name=f"bn_conv_{block_id}_{block_id}")(conv6)
                layers[dict_key].append(conv6)

            prev_layer = conv6

        return layers

    def __build_cells_regressors(self, start_layer, initial_block_id: int, n_filters: int, dilation_rate: int):
        """ Builds the regressor task.

        This branch of the neural network is an addition to the original U-Net architecture. The main goal of this
        branch is to obtain the number of cells on an image. To do so is based on the regression principles, using
        special kind of loss function to do it.

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
        last_layer = keras_layer.Conv2D(n_filters * (2 ** 3), (3, 3), dilation_rate=dilation_rate, activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal', name=f"conv_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Conv2D(n_filters * (2 ** 3), (3, 3), dilation_rate=dilation_rate, activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        name=f"conv_{block_id}_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name=f"mp_{block_id}")(last_layer)
        layers.append(last_layer)

        block_id = initial_block_id + 2
        last_layer = keras_layer.Conv2D(n_filters * (2 ** 4), (3, 3), dilation_rate=dilation_rate, activation='relu',
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        name=f"conv_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = keras_layer.Conv2D(n_filters * (2 ** 4), (3, 3), dilation_rate=dilation_rate, activation='relu',
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

    def build_unet(self, n_filters=16, dilation_rate: int = 1, n_blocks: int = 4):
        """ Builds the graph and model for the U-Net.

        The U-Net, first introduced by Ronnenberger et al., is an encoder-decoder architecture.
        Build through the stack of 2D convolutional and up sampling 2D.

        Args:
            n_filters:
            dilation_rate:
            n_blocks:
        """
        # Define input batch shape
        input_image = keras_layer.Input(self.__input_size, name="input_image")
        encoder, last_block_id = self.__build_encoder(n_filters=n_filters, start_layer=input_image, n_blocks=n_blocks,
                                                      dilation_rate=dilation_rate)

        if self.__build_regressor:
            regressor, last_block_id = self.__build_cells_regressors(start_layer=list(encoder.values())[-1][-1],
                                                                     initial_block_id=6,
                                                                     n_filters=n_filters, dilation_rate=dilation_rate)

        filters_size = [n_filters * (2 ** i) for i in range(0, n_blocks)]
        filters_size = filters_size[::-1]

        decoder = self.__build_decoder(encoder=encoder, dilation_rate=dilation_rate,
                                       initial_block_id=last_block_id + 1,
                                       filters=filters_size)

        if self.__n_channels == 1:
            last_activation = "sigmoid"
        else:
            last_activation = "softmax"

        conv10 = keras_layer.Conv2D(self.__n_channels, (1, 1), activation=last_activation, padding='same',
                                    dilation_rate=dilation_rate, kernel_initializer='he_normal', name="img_out")(
            list(decoder.values())[-1][-1])

        if not self.__build_rpn and self.__build_regressor:
            model = keras_model.Model(inputs=input_image, outputs=[conv10, regressor[-1]])
        elif not self.__build_rpn:
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
            input_rpn_match = keras_layer.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = keras_layer.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # RPN Loss
            rpn_class_loss = keras_layer.Lambda(lambda x: rpn_model.class_loss_graph(*x),
                                                name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = keras_layer.Lambda(lambda x: rpn_model.bbox_loss_graph(config, *x),
                                               name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])

            # Input of the model
            inputs = [input_image, input_rpn_match, input_rpn_bbox]

            # Output of the model
            outputs = [conv10,
                       rpn_class_logits,
                       rpn_class,
                       rpn_bbox,
                       rpn_class_loss,
                       rpn_bbox_loss]

            model = keras_model.Model(inputs, outputs, name='r-unet')

        self.__internal_model = model

    def compile(self, loss_func: Union[str, Callable] = "categorical_crossentropy", check: bool = False):
        """ Compiles the models.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.
            check (bool): Only used in the RPN context. If true checks

        Returns:

        """
        if not self.__build_rpn:
            loss_functions = {"img_out": loss_func}

            if self.__build_regressor:
                loss_functions['regressor_output'] = 'mean_absolute_error'

            self.__internal_model.compile(optimizer=Adam(lr=3e-5), loss=loss_functions,
                                          metrics=['categorical_accuracy'])
        else:
            loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

            for name in loss_names:
                layer = self.__internal_model.get_layer(name)
                if not check and layer.output in self.__internal_model.losses:
                    continue
                loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * 1.0)
                self.__internal_model.add_loss(loss)

            self.__internal_model.compile(optimizer=Adam(lr=self.__config.LEARNING_RATE),
                                          loss=[loss_func, None, None, None, None, None])

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None):
        """ Trains the model with the info passed as parameters.

        The keras model is trained with the information passed as parameters. The info is defined
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
        if self.__is_trained:
            warnings.warn("Model already trained, starting new training")

        if callbacks is None:
            callbacks = []

        if check_point_path is not None:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                   save_weights_only=False, save_best_only=True))

        if val_generator is not None:
            self.__internal_model.fit(train_generator, validation_data=val_generator, epochs=epochs,
                                      validation_steps=validation_steps, callbacks=callbacks,
                                      steps_per_epoch=steps_per_epoch)
        else:
            self.__internal_model.fit(train_generator, epochs=epochs, callbacks=callbacks,
                                      steps_per_epoch=steps_per_epoch)
        self.__is_trained = True

    @property
    def model(self):
        return self.__internal_model
