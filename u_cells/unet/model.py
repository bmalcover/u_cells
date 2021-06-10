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
from typing import Callable, Union, Tuple
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
                 batch_normalization: bool, config_net: config.Config = None, rpn: bool = False):

        self.__input_size: Tuple[int, int, int] = input_size
        self.__batch_normalization: bool = batch_normalization
        self.__build_rpn: bool = rpn
        self.__n_channels: int = out_channel
        self.__config = config_net

        self.__keras_model = None
        self.__is_trained = False

    def build_unet(self, n_filters=16, dilation_rate=1):
        """ Builds the graph and model for the U-Net.

        The U-Net, first introduced by Ronnenberger et al., is an encoder-decoder architecture.
        Build through the stack of 2D convolutional and up sampling 2D.

        Args:
            n_filters:
            dilation_rate:
        """
        bn: bool = self.__batch_normalization

        # Define input batch shape
        input_image = KL.Input(self.__input_size, name="input_image")

        conv1 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_1")(input_image)
        if bn:
            conv1 = KL.BatchNormalization(name="bn_conv_1")(conv1)

        conv1 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_1_1")(
            conv1)

        if bn:
            conv1 = KL.BatchNormalization(name="bn_conv_1_1")(conv1)

        pool1 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='mp_1')(conv1)

        conv2 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_2")(
            pool1)
        if bn:
            conv2 = KL.BatchNormalization(name="bn_conv_2")(conv2)

        conv2 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_2_2")(
            conv2)
        if bn:
            conv2 = KL.BatchNormalization(name="bn_conv_2_2")(conv2)

        pool2 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name="mp_2")(conv2)

        conv3 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_3")(
            pool2)
        if bn:
            conv3 = KL.BatchNormalization(name="bn_conv_3")(conv3)

        conv3 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_3_3")(
            conv3)

        if bn:
            conv3 = KL.BatchNormalization(name="bn_conv_3_3")(conv3)

        pool3 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name="mp_3")(conv3)

        conv4 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_4")(
            pool3)
        if bn:
            conv4 = KL.BatchNormalization(name="bn_conv_4")(conv4)

        conv4 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_4_4")(
            conv4)

        if bn:
            conv4 = KL.BatchNormalization(name="bn_conv_4_4")(conv4)

        pool4 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name="mp_4")(conv4)

        conv5 = KL.Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_5")(
            pool4)
        if bn:
            conv5 = KL.BatchNormalization(name="bn_conv5")(conv5)

        conv5 = KL.Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_5_5")(
            conv5)
        if bn:
            conv5 = KL.BatchNormalization(name="bn_conv5_5")(conv5)

        up6 = KL.concatenate([KL.Conv2D(n_filters * 8, (2, 2), activation='relu', padding='same',
                                        dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                        name="conv_6")(
            KL.UpSampling2D(size=(2, 2), name="up_6")(conv5)), conv4], name="conct_6", axis=3)

        conv6 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_6_6")(
            up6)
        if bn:
            conv6 = KL.BatchNormalization(name="bn_conv_6")(conv6)

        conv6 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_6_6_6")(conv6)
        if bn:
            conv6 = KL.BatchNormalization(name="bn_conv_6_6")(conv6)

        up7 = KL.concatenate([KL.Conv2D(n_filters * 4, (2, 2), activation='relu', padding='same',
                                        dilation_rate=dilation_rate, kernel_initializer='he_normal',
                                        name="conv_7")(
            KL.UpSampling2D(size=(2, 2), name="up_7")(conv6)), conv3], axis=3, name='conc_7')

        conv7 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_7_7")(
            up7)
        if bn:
            conv7 = KL.BatchNormalization(name="bn_conv_7")(conv7)

        conv7 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal',
                          name="conv_7_7_7")(conv7)
        if bn:
            conv7 = KL.BatchNormalization(name="bn_conv_7_7")(conv7)

        up8 = KL.concatenate([KL.Conv2D(n_filters * 2, (2, 2), activation='relu', padding='same',
                                        dilation_rate=dilation_rate,
                                        kernel_initializer='he_normal')(
            KL.UpSampling2D(size=(2, 2))(conv7)), conv2], axis=3, name='conc_8')

        conv8 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal')(up8)
        if bn:
            conv8 = KL.BatchNormalization()(conv8)

        conv8 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal')(conv8)
        if bn:
            conv8 = KL.BatchNormalization()(conv8)

        up9 = KL.concatenate([KL.Conv2D(n_filters * 8, (2, 2), activation='relu', padding='same',
                                        dilation_rate=dilation_rate,
                                        kernel_initializer='he_normal')(
            KL.UpSampling2D(size=(2, 2))(conv8)), conv1], axis=3, name='conc_9')

        conv9 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate)(up9)
        if bn:
            conv9 = KL.BatchNormalization()(conv9)

        conv9 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same',
                          dilation_rate=dilation_rate, kernel_initializer='he_normal')(conv9)
        if bn:
            conv9 = KL.BatchNormalization()(conv9)

        if self.__n_channels == 1:
            last_activation = "sigmoid"
        else:
            last_activation = "softmax"

        conv10 = KL.Conv2D(self.__n_channels, (1, 1), activation=last_activation, padding='same',
                           dilation_rate=dilation_rate, kernel_initializer='he_normal')(conv9)

        if not self.__build_rpn:
            model = KM.Model(inputs=input_image, outputs=conv10)
        else:
            if config is None:
                raise AttributeError("Config for RPN model not defined")
            # We connect the U-Net to the RPN via the last CONV5 layer, the last layer of the
            # decoder.
            rpn = rpn_model.build_rpn_model(depth=n_filters * 16)  # Conv5
            rpn_output = rpn([conv5])

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
            self.__keras_model.compile(optimizer=Adam(lr=3e-5), loss=loss_func,
                                       metrics=['categorical_accuracy'])
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

            self.__keras_model.compile(optimizer=Adam(lr= self.__config.LEARNING_RATE),
                                       loss=[loss_func, None, None, None, None, None])

    def train(self, train_generator, val_generator, epochs: int, steps_per_epoch: int,
              validation_steps: int, check_point_path: Union[str, None], callbacks=None):
        if self.__is_trained:
            warnings.warn("Model already trained, starting new training")

        if callbacks is None:
            callbacks = []

        if check_point_path is None:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(check_point_path, verbose=0,
                                                   save_weights_only=False, save_best_only=True))

        self.__keras_model.fit(train_generator, validation_data=val_generator, epochs=epochs,
                               validation_steps=validation_steps, callbacks=callbacks,
                               steps_per_epoch=steps_per_epoch)

        self.__is_trained = True

    @property
    def model(self):
        return self.__keras_model
