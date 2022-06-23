# -*- coding: utf-8 -*-
""" Suite of tests for the delta decoder layer.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
from unittest import TestCase

import tensorflow as tf
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_models

from u_rpn import layers as rpn_layers


class TestDeltaDecoder(TestCase):
    """Suite of tests for the delta decoding layer.

    Tests:
        - Shape of the output of the layer with a batch size of 1.
        - Shape of the output of the layer with a batch size of 10.
    """

    def __init__(self, *args: list, **kwargs: dict):
        super(TestDeltaDecoder, self).__init__(*args, **kwargs)

        self.__anchors = tf.ones((1, 200, 4))

    def __build_test_model(self, size: int = 80) -> keras_models.Model:
        """Builds a model with the delta decoding layer.

        To be able to test the behaviour of the layer, we need to build a model with the delta
        decoding layer. In this case we build a model with the two needed inputs and the output of
        the layer.

        Args:
            size(int): Maximum number of bounding boxes.
        Returns:
            keras_models.Model: Model with the delta decoding layer.
        """
        deltas = keras_layers.Input(shape=(None, 4))
        obj_ness = keras_layers.Input(shape=(None, 2))

        out = rpn_layers.DeltaDecoder(self.__anchors, size, (512, 512))(deltas, obj_ness)
        model = keras_models.Model([deltas, obj_ness], out)

        return model

    def test_shape_b1(self) -> None:
        """Test the shape of the output of the layer with a batch size of 1."""
        model = self.__build_test_model()

        output = model.predict([tf.ones((1, 200, 4)), tf.ones((1, 200, 2))])

        self.assertAlmostEqual(output.shape[1], 80)

    def test_shape_b10(self) -> None:
        """Test the shape of the output of the layer with a batch size of 10."""
        model = self.__build_test_model()

        output = model.predict([tf.ones((10, 200, 4)), tf.ones((10, 200, 2))])

        self.assertAlmostEqual(output.shape[1], 80)
        self.assertAlmostEqual(output.shape[0], 10)
