# -*- coding: utf-8 -*-
""" Test suite for segmentations loss functions.

Written by: Miquel Miró Nicolau (UIB)
"""
from unittest import TestCase

import numpy as np
import tensorflow as tf

from u_rpn.losses import segmentation


class TestWeightedBCE(TestCase):
    """ Test suite for weighted binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with half of the inputs equals to 1.
    """

    def test_equals_nulls(self):
        pred = tf.constant(np.zeros((10, 10)))
        target = tf.constant(np.zeros((10, 10)))

        self.assertAlmostEqual(segmentation.bce_weighted_rpn(pred, None, target).numpy(), 0)

    def test_equals_positives(self):
        pred = tf.constant(np.ones((10, 10)))
        target = tf.constant(np.ones((10, 10)))

        self.assertAlmostEqual(segmentation.bce_weighted_rpn(pred, None, target).numpy(), 0)

    def test_equals_multiple(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        pred[0:2, 0:2] = 1
        target[0:2, 0:2] = 1

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(segmentation.bce_weighted_rpn(pred, None, target).numpy(), 0)

