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
from typing import Callable, Union

import numpy as np 
import os
import rpn

from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import losses
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.layers as KE
import tensorflow as tf


def multiclass_weighted_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted Dice loss.
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute weighted Dice loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * keras.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)
	
        iou = 1 - numerator / denominator
	
        return iou

    return loss
    
def dice_coef_loss(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    
    From: TensorLayer

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> outputs = tl.act.pixel_wise_softmax(outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

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


def build_unet(n_filters=16, bn=True, dilation_rate=1, input_size=(256, 256, 1),
               output_channels=3, loss_func="categorical_crossentropy", use_rpn: bool = False):
    '''Validation Image data generator
        Inputs: 
            n_filters - base convolution filters
            bn - flag to set batch normalization
            dilation_rate - convolution dilation rate
        Output: Unet keras Model
    '''
    # Define input batch shape
    input_image = KL.Input(input_size)

    conv1 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_1")(input_image)
    if bn:
        conv1 = KL.BatchNormalization(name="bn_conv_1")(conv1)
        
    conv1 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_1_1")(conv1)

    if bn:
        conv1 = KL.BatchNormalization(name="bn_conv_1_1")(conv1)

    pool1 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='mp_1')(conv1)

    conv2 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_2")(pool1)
    if bn:
        conv2 = KL.BatchNormalization(name="bn_conv_2")(conv2)
        
    conv2 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_2_2")(conv2)
    if bn:
        conv2 = KL.BatchNormalization(name="bn_conv_2_2")(conv2)

    pool2 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name="mp_2")(conv2)

    conv3 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_3")(pool2)
    if bn:
        conv3 = KL.BatchNormalization(name="bn_conv_3")(conv3)
        
    conv3 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_3_3")(conv3)

    if bn:
        conv3 = KL.BatchNormalization(name="bn_conv_3_3")(conv3)

    pool3 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name="mp_3")(conv3)

    conv4 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_4")(pool3)
    if bn:
        conv4 = KL.BatchNormalization(name="bn_conv_4")(conv4)
        
    conv4 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_4_4")(conv4)

    if bn:
        conv4 = KL.BatchNormalization(name="bn_conv_4_4")(conv4)

    pool4 = KL.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name="mp_4")(conv4)

    conv5 = KL.Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_5")(pool4)
    if bn:
        conv5 = KL.BatchNormalization(name="bn_conv5")(conv5)
        
    conv5 = KL.Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_5_5")(conv5)
    if bn:
        conv5 = KL.BatchNormalization(name="bn_conv5_5")(conv5)
        
    up6 = KL.concatenate([KL.Conv2D(n_filters * 8, (2 ,2), activation='relu', padding='same',
                 dilation_rate=dilation_rate, kernel_initializer = 'he_normal', name="conv_6")(KL.UpSampling2D(size=(2, 2), name="up_6")(conv5)), conv4], name = "conct_6" ,axis = 3)
    
    conv6 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_6_6")(up6)
    if bn:
        conv6 = KL.BatchNormalization(name="bn_conv_6")(conv6)
        
    conv6 = KL.Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_6_6_6")(conv6)
    if bn:
        conv6 = KL.BatchNormalization(name="bn_conv_6_6")(conv6)
        
    up7 = KL.concatenate([KL.Conv2D(n_filters * 4, (2 ,2), activation='relu', padding='same',
                 dilation_rate=dilation_rate, kernel_initializer = 'he_normal', name="conv_7")(KL.UpSampling2D(size=(2, 2), name="up_7")(conv6)), conv3], axis=3, name = 'conc_7')
    
    conv7 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name="conv_7_7")(up7)
    if bn:
        conv7 = KL.BatchNormalization(name="bn_conv_7")(conv7)
        
    conv7 = KL.Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal', name = "conv_7_7_7")(conv7)
    if bn:
        conv7 = KL.BatchNormalization(name="bn_conv_7_7")(conv7)
       
    up8 = KL.concatenate([KL.Conv2D(n_filters * 2, (2 ,2), activation='relu', padding='same',
                 dilation_rate=dilation_rate, kernel_initializer = 'he_normal')(KL.UpSampling2D(size=(2, 2))(conv7)), conv2], axis=3, name = 'conc_8')
    
    conv8 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal')(up8)
    if bn:
        conv8 = KL.BatchNormalization()(conv8)
        
    conv8 = KL.Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal')(conv8)
    if bn:
        conv8 = KL.BatchNormalization()(conv8)
       
    up9 = KL.concatenate([KL.Conv2D(n_filters * 8, (2 ,2), activation='relu', padding='same',
                 dilation_rate=dilation_rate, kernel_initializer = 'he_normal')(KL.UpSampling2D(size=(2, 2))(conv8)), conv1], axis=3, name = 'conc_9')
    
    conv9 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
    if bn:
        conv9 = KL.BatchNormalization()(conv9)
        
    conv9 = KL.Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal')(conv9)
    if bn:
        conv9 = KL.BatchNormalization()(conv9)
    
    if output_channels == 1:
        last_activation = "sigmoid"
    else:
        last_activation = "softmax"
        
    conv10 = KL.Conv2D(output_channels, (1, 1), activation=last_activation, padding = 'same', dilation_rate = dilation_rate, kernel_initializer = 'he_normal')(conv9)
    
    if not use_rpn:
        model = KM.Model(inputs=input_image, outputs=conv10)
        model.compile(optimizer=Adam(lr=3e-5), loss=loss_func, metrics=['categorical_accuracy'])
    else:
        ####################################################################################################################
        #####                                                RPN MODEL                                                 #####
        ####################################################################################################################
    
        # We connect the U-Net to the RPN via the last CONV5 layer, the last layer of the decoder.
        rpn = build_rpn_model(depth = n_filters * 16) # Conv5
        rpn_output = rpn([conv5])

        # RPN Output
        rpn_class_logits, rpn_class, rpn_bbox = rpn_output

        # RPN GT
        input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # RPN Loss
        rpn_class_loss = KL.Lambda(lambda x: rpn.class_loss_graph(*x), name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: rpn.bbox_loss_graph(config, *x), name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])

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

        # These two losses can not be passed as default loss because they do not accept y_true, y_pred
        # for this reason are added via add_loss. 
        loss_names = ["rpn_class_loss",  "rpn_bbox_loss"]

        for name in loss_names:
            layer = model.get_layer(name)
            if layer.output in model.losses:
                continue
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * 1.0)
            model.add_loss(loss)

        model.compile(optimizer=Adam(lr=3e-5), loss=[loss_func, None, None, None, None, None])

    return model
    
