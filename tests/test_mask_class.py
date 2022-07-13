""" Suite of tests for Mask-Class branch.

Tests:
    - Test shape. Tests whether the output shape of the Mask-Class branch is correct.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_model

from u_rpn.branches import MaskClass


class TestMaskClass(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        inputs = keras_layers.Input(shape=(None, None, 10))
        mask_class = MaskClass(10, 10)(inputs)
        self._model = keras_model.Model(inputs=inputs, outputs=[mask_class])

    def test_shape(self) -> None:
        """Tests whether the shape is the one expect from the merge mask branch.

        Args:
            None
        Returns:
            None
        """
        tensor1 = tf.constant(np.zeros((1, 10, 10, 10)))

        out = self._model(tensor1).numpy()
        self.assertTupleEqual(out.shape, (1, 10))
