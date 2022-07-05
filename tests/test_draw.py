""" Suite of tests for the Draw Bbox layer.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
from typing import Tuple
from unittest import TestCase

import tensorflow as tf
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_models

from u_rpn import layers as rpn_layers


class TestDrawBboxes(TestCase):
    """Suite of tests for the Draw Boxes layer.

    Tests:
        - Shape of the output of the layer with a batch size of 1.
        - Shape of the output of the layer with a batch size of 10.
    """

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)

        self.__anchors = tf.ones((1, 200, 4))

    @staticmethod
    def __build_test_model(image_size: Tuple[int, int, int, int]) -> keras_models.Model:
        """Builds a model with the delta decoding layer.

        To be able to test the behaviour of the layer, we need to build a model with the delta
        decoding layer. In this case we build a model with the two needed inputs and the output of
        the layer.

        Args:
            image_size(int): Maximum number of bounding boxes.
        Returns:
            keras_models.Model: Model with the delta decoding layer.
        """
        bboxes = keras_layers.Input(shape=(image_size[-1], 4))

        out = rpn_layers.DrawBoxes(image_size)(bboxes)
        model = keras_models.Model(bboxes, out)

        return model

    def test_shape_b1(self) -> None:
        """Test the shape of the output of the layer with a batch size of 1."""
        model = TestDrawBboxes.__build_test_model((6, 128, 128, 80))

        output = model.predict(tf.ones((6, 80, 4)))

        self.assertAlmostEqual(output.shape[-1], 80)

    def test_shape_b10(self) -> None:
        """Test the shape of the output of the layer with a batch size of 10."""
        model = TestDrawBboxes.__build_test_model((6, 128, 128, 80))
        output = model.predict(tf.ones((6, 80, 4)))

        self.assertAlmostEqual(output.shape[-1], 80)
        self.assertAlmostEqual(output.shape[0], 6)
