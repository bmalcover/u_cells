""" Suite of tests for the Mask Bbox layer.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
from typing import Tuple
from unittest import TestCase

import tensorflow as tf
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_models

from u_rpn import layers as rpn_layers


class TestMaskBboxes(TestCase):
    """Suite of tests:
    - Test mask shape
    - Test whether the wanted píxels are drawn.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def __build_test_model(image_size: Tuple[int, int]) -> keras_models.Model:
        """Builds model with the MaskBboxes layer.

        Args:
            image_size: Two int tuple with the image size.

        Returns:
            keras_models.Model: Model with the MaskBboxes layer.
        """
        bboxes = keras_layers.Input(shape=(None, None, 4))

        out = rpn_layers.MaskBboxes(image_size=image_size)(bboxes)
        model = keras_models.Model(bboxes, out)

        return model

    def test_bbox_shape(self) -> None:
        """Unit test for the shape of the output of the layer.

        Returns:
            None
        """
        model = TestMaskBboxes.__build_test_model((10, 10))

        bbox = tf.convert_to_tensor([[[0.1, 0.1, 0.2, 0.2], [0.1, 0.2, 0.5, 0.3]]])

        res = model.predict(bbox)

        self.assertTupleEqual(res.shape, (1, 10, 10, 2))

    def test_draw(self) -> None:
        """Unit test for the drawing of the bounding boxes.

        Returns:
            None
        """
        model = TestMaskBboxes.__build_test_model((10, 10))

        bbox = tf.convert_to_tensor([[[0.2, 0.2, 0.3, 0.5], [0.2, 0.4, 0.4, 0.7]]])

        res = model.predict(bbox)

        first_px = res[0, :, :, 0][2][2]

        self.assertAlmostEqual(first_px, 1)
