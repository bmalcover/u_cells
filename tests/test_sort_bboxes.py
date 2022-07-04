""" Suite of tests for the Sort Bbox layer.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
from unittest import TestCase

import tensorflow as tf
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_models

from u_rpn import layers as rpn_layers


class TestSortBboxes(TestCase):
    """Suite of test for the Sort Bbox layer.

    Tests:
        - Shape of the output of the layer with a batch size of 5.
    """

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)

    @staticmethod
    def __build_test_model() -> keras_models.Model:
        """Builds a model with the sort bboxes layer.

        To be able to test the behaviour of the layer, we need to build a model to accept the input
        of the layer. In this case we build a model with the inputs.

        Returns:
            keras_models.Model: Model with the sort bboxes layer.
        """
        bboxes = keras_layers.Input(shape=(None, 4))

        out = rpn_layers.SortBboxes()(bboxes)
        model = keras_models.Model(bboxes, out)

        return model

    def test_shape(self) -> None:
        """Test the shape of the output of the layer with a batch size of 5.

        Returns:
            None: Nothing.
        """
        test_model = self.__build_test_model()

        bboxes = tf.zeros((5, 10, 4))
        bboxes = tf.concat([bboxes, tf.ones((5, 1, 4))], axis=1)

        res = test_model.predict(bboxes)

        self.assertTupleEqual(res.shape, bboxes.numpy().shape)
