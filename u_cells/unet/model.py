# -*- coding: utf-8 -*-
""" Module containing all functions to build the U-Net model.

This module contains the set of functions that defines the original U-Net networks.
This network was proposed by Ronnenberger et al. and is based on a Encoder-Decoder 
architecture.

Furthermore also contains the loss functions and a set of modification to the original
work to be able to work with bounding boxes.

Loss functions:
    Dice coefficient
    Sorensen coefficient
    Jaccard coefficient
"""
from typing import Callable, Union, Tuple, List
import warnings

import numpy as np

from tensorflow.keras.optimizers import *
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow as tf

from u_cells.common import config
from u_cells.rpn import model as rpn_model


def multiclass_weighted_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[
    [tf.Tensor, tf.Tensor], tf.Tensor]:
    """ Weighted Dice loss.

    Used as loss function for multi-class image segmentation with one-hot encoded masks.

    Args:
        class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor],
                        len=<N_CLASSES>)
    Returns:
        Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Compute weighted Dice loss.


        Args:
            y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>,
                                                  <N_CLASSES>))
            y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>,
                                                       <N_CLASSES>))

        Returns:
            Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights  # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)

        iou = 1 - numerator / denominator

        return iou

    return loss


def dice_coef_loss(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5):
    """ Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    From: TensorLayer

    Args:
        output (Tensor): A distribution with shape: [batch_size, ....], (any dimensions).
        target (Tensor): The target distribution, format the same with `output`.
        loss_type (str): ``jaccard`` or ``sorensen``, default is ``sorensen``.
        axis (tuple | int | None): All dimensions are reduced, default ``[1,2,3]``.
        smooth (float):  This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
            dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to
            0 (even the image values lower than the threshold), so in this case, higher smooth
            can have a higher dice.

    Returns:
    References:
        `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`
    """

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')

    return 1 - dice


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

        self.__keras_model = None
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

        """
        bn: bool = self.__batch_normalization

        layers = {}
        prev_layer = start_layer
        for i in range(initial_block_id, initial_block_id + n_blocks):
            dict_key = name + "_" + str(i + 1)
            layers[dict_key] = []

            n_filters_layer = n_filters * (2 ** i)

            conv1 = KL.Conv2D(n_filters_layer, (3, 3), activation='relu', padding='same',
                              dilation_rate=dilation_rate, kernel_initializer='he_normal',
                              name=f"conv_{i + 1}")(prev_layer)
            layers[dict_key].append(conv1)

            if bn:
                conv1 = KL.BatchNormalization(name=f"bn_conv_{i + 1}")(conv1)
                layers[dict_key].append(conv1)

            conv1 = KL.Conv2D(n_filters_layer, (3, 3), activation='relu', padding='same',
                              dilation_rate=dilation_rate, kernel_initializer='he_normal',
                              name=f"conv_{i + 1}_{i + 1}")(conv1)
            layers[dict_key].append(conv1)

            if bn:
                conv1 = KL.BatchNormalization(name=f"bn_conv_{i + 1}_{i + 1}")(conv1)
                layers[dict_key].append(conv1)

            if (i + 1) != initial_block_id + n_blocks:
                pool1 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last',
                                        name=f"mp_{i + 1}")(conv1)
                layers[dict_key].append(pool1)

                prev_layer = pool1

        return layers

    def __build_decoder(self, encoder, filters: List[int], name: str = "decode",
                        dilation_rate: int = 1, initial_block_id: int = 0):
        bn: bool = self.__batch_normalization

        layers = {}
        prev_layer = list(encoder.values())[-1][-1]

        encoder_layers = list(encoder.values())[:-1]
        encoder_layers = encoder_layers[::-1]
        for i, (filter_per_layer, enc_layer) in enumerate(zip(filters, encoder_layers)):
            block_id: int = initial_block_id + i
            dict_key = name + "_" + str(block_id)

            layers[dict_key] = []

            up6 = KL.concatenate(
                [KL.Conv2D(filter_per_layer, (2, 2), activation='relu', padding='same',
                           dilation_rate=dilation_rate, kernel_initializer='he_normal',
                           name=f"conv_{block_id}")(
                    KL.UpSampling2D(size=(2, 2), name=f"up_{block_id}")(prev_layer)),
                    enc_layer[-2]], name=f"conct_{block_id}", axis=3)
            layers[dict_key].append(up6)

            conv6 = KL.Conv2D(filter_per_layer, (3, 3), activation='relu', padding='same',
                              dilation_rate=dilation_rate, kernel_initializer='he_normal',
                              name=f"conv_{block_id}_{block_id}")(up6)
            layers[dict_key].append(conv6)

            if bn:
                conv6 = KL.BatchNormalization(name=f"bn_conv_{block_id}")(conv6)
                layers[dict_key].append(conv6)

            conv6 = KL.Conv2D(filter_per_layer, (3, 3), activation='relu', padding='same',
                              dilation_rate=dilation_rate, kernel_initializer='he_normal',
                              name=f"conv_{block_id}_{block_id}_{block_id}")(conv6)
            layers[dict_key].append(conv6)

            if bn:
                conv6 = KL.BatchNormalization(name=f"bn_conv_{block_id}_{block_id}")(conv6)
                layers[dict_key].append(conv6)

            prev_layer = conv6

        return layers

    def __build_cells_regressors(self, start_layer, initial_block_id: int, n_filters: int, dilation_rate: int):
        layers = []
        last_layer = start_layer

        block_id = initial_block_id + 1
        last_layer = KL.Conv2D(n_filters * (2 ** 3), (3, 3), dilation_rate=dilation_rate, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal', name=f"conv_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = KL.Conv2D(n_filters * (2 ** 3), (3, 3), dilation_rate=dilation_rate, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal',
                               name=f"conv_{block_id}_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name=f"mp_{block_id}")(last_layer)
        layers.append(last_layer)

        block_id = initial_block_id + 2
        last_layer = KL.Conv2D(n_filters * (2 ** 4), (3, 3), dilation_rate=dilation_rate, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal',
                               name=f"conv_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = KL.Conv2D(n_filters * (2 ** 4), (3, 3), dilation_rate=dilation_rate, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal',
                               name=f"conv_{block_id}_{block_id}")(last_layer)
        layers.append(last_layer)

        last_layer = KL.Dense(1024, name="dense_1")(last_layer)
        layers.append(last_layer)

        last_layer = KL.Dense(1024, name="dense_2")(last_layer)
        layers.append(last_layer)

        last_layer = KL.Dense(1, name="regressor_output")(last_layer)
        layers.append(last_layer)

        return layers, block_id

    def build_unet(self, n_filters=16, dilation_rate: int = 1):
        """ Builds the graph and model for the U-Net.

        The U-Net, first introduced by Ronnenberger et al., is an encoder-decoder architecture.
        Build through the stack of 2D convolutional and up sampling 2D.

        Args:
            n_filters:
            dilation_rate:
        """
        # Define input batch shape
        input_image = KL.Input(self.__input_size, name="input_image")
        encoder = self.__build_encoder(n_filters=n_filters, start_layer=input_image, n_blocks=4,
                                       dilation_rate=dilation_rate)

        if self.__build_regressor:
            regressor, last_block_id = self.__build_cells_regressors(start_layer=list(encoder.values())[-1][-1],
                                                                     initial_block_id=6,
                                                                     n_filters=n_filters, dilation_rate=dilation_rate)

        decoder = self.__build_decoder(encoder=encoder, dilation_rate=dilation_rate,
                                       initial_block_id=last_block_id + 1,
                                       filters=[n_filters * 8, n_filters * 4, n_filters * 2,
                                                n_filters * 1])

        if self.__n_channels == 1:
            last_activation = "sigmoid"
        else:
            last_activation = "softmax"

        conv10 = KL.Conv2D(self.__n_channels, (1, 1), activation=last_activation, padding='same',
                           dilation_rate=dilation_rate, kernel_initializer='he_normal', name="img_out")(
            list(decoder.values())[-1][-1])

        if not self.__build_rpn and self.__build_regressor:
            model = KM.Model(inputs=input_image, outputs=[conv10, regressor[-1]])
        elif not self.__build_rpn:
            model = KM.Model(inputs=input_image, outputs=conv10)
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
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # RPN Loss
            rpn_class_loss = KL.Lambda(lambda x: rpn_model.class_loss_graph(*x),
                                       name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_model.bbox_loss_graph(config, *x),
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

            model = KM.Model(inputs, outputs, name='r-unet')

        self.__keras_model = model

    def compile(self, loss_func: Union[str, Callable] = "categorical_crossentropy"):
        """ Compiles the models.

        This function has two behaviors depending on the inclusion of the RPN. In the case of
        vanilla U-Net this function works as wrapper for the keras.model compile method.

        Args:
            loss_func (str | Callable): Loss function to apply to the main output of the U-Net.

        Returns:

        """
        if not self.__build_rpn:
            loss_functions = {"img_out": loss_func}

            if self.__build_regressor:
                loss_functions['regressor_output'] = 'mean_absolute_error'

            self.__keras_model.compile(optimizer=Adam(lr=3e-5), loss=loss_functions, metrics=['categorical_accuracy'])
        else:
            # These two losses can not be passed as default loss because they do not accept y_true,
            # y_pred for this reason are added via add_loss.

            loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

            for name in loss_names:
                layer = self.__keras_model.get_layer(name)
                if layer.output in self.__keras_model.losses:
                    continue
                loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * 1.0)
                self.__keras_model.add_loss(loss)

            self.__keras_model.compile(optimizer=Adam(lr=self.__config.LEARNING_RATE),
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
            self.__keras_model.fit(train_generator, validation_data=val_generator, epochs=epochs,
                                   validation_steps=validation_steps, callbacks=callbacks,
                                   steps_per_epoch=steps_per_epoch)
        else:
            self.__keras_model.fit(train_generator, epochs=epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
        self.__is_trained = True

    @property
    def model(self):
        return self.__keras_model
