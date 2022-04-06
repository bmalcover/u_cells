# -*- coding: utf-8 -*-
""" Test suites for U-RPN network

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import unittest

import tensorflow.keras.layers as keras_layer

from u_rpn import model as u_model
from u_rpn import configurations as u_configs
from u_rpn.data import datasets
from u_rpn.data import rpn as u_rpn_data


class TestRPN(unittest.TestCase):
    @staticmethod
    def __build_model(mode: u_model.rpn.NeuralMode, config, training=None):
        encoder = u_model.unet.EncoderUNet(input_size=(512, 512, 3))
        input_image, embedded = encoder.build(n_filters=16, last_activation='softmax',
                                              dilation_rate=1, layer_depth=5, training=training)

        features = list(encoder.layers.values())[-2]
        features = keras_layer.Concatenate(axis=-1, name="conc_1")([features,
                                                                    keras_layer.Conv2DTranspose(
                                                                        128, kernel_size=(3, 3),
                                                                        strides=(2, 2),
                                                                        name="convd_tranposed_1",
                                                                        padding="same")(list(
                                                                        encoder.layers.values())[
                                                                                            -1])])
        features = keras_layer.Conv2D(256, (1, 1), name="conv_1")(features)
        features = keras_layer.Concatenate(axis=-1, name="conc_2")(
            [(list(encoder.layers.values())[-3]),
             keras_layer.Conv2DTranspose(256, name="convd_tranposed_2", kernel_size=(3, 3),
                                         strides=(2, 2), padding="same")(features)])
        features = keras_layer.Conv2D(256, (1, 1), name="conv_2")(features)

        rpn = u_model.rpn.RPN(mode, (512, 512, 3), features, 256, None,
                              input_image, config)
        rpn_out, rpn_conv = rpn.build_rpn(features)

        _, rpn_class, rpn_bbox = rpn_out

        decoder = u_model.unet.DecoderUNet(input_size=None, residual=True,
                                           n_channels=config.MAX_GT_INSTANCES,
                                           class_output_size=128)
        mask_out, class_out = decoder.build(n_filters=16, last_activation='sigmoid',
                                            encoder=encoder, dilation_rate=1, embedded=embedded,
                                            training=training)

        rpn.build(mask_shape=[None, None, None], rpn=rpn_out, mask_output=mask_out,
                  do_mask=True, mask_class=class_out)

        return rpn

    def test_build_train_net(self):
        """ Test if the model is built correctly without exceptions. """
        self.__build_model(u_model.rpn.NeuralMode.TRAIN, config=u_configs.CellConfig())

    def test_build_always_train_net(self):
        """ Test if the model is built correctly without exceptions when training always True."""
        self.__build_model(u_model.rpn.NeuralMode.TRAIN, training=True,
                           config=u_configs.CellConfig())

    def test_one_epoch(self):
        """Test if the model is able to execute one step of one epoch"""
        config = u_configs.CellConfig()
        config.STEPS_PER_EPOCH = 1

        model = self.__build_model(u_model.rpn.NeuralMode.TRAIN, training=True, config=config)

        dataset = datasets.ErithocytesPreDataset("./in/test_data", "data.json",
                                                 divisor=255)

        generator = u_rpn_data.DataGenerator(1, dataset, pre_calculated=True,
                                             config=config,
                                             phantom_output=True, shuffle=False,
                                             size_anchors=dataset.anchors)

        model.compile(do_mask=True, do_class_mask=True)
        model.train(train_generator=generator, val_generator=None, validation_steps=2, epochs=1,
                    check_point_path=None)
