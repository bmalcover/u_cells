# -*- coding: utf-8 -*-
""" Test suite for segmentations bce loss functions.

Written by: Miquel Miró Nicolau (UIB)
"""
from unittest import TestCase

import numpy as np
import tensorflow as tf
import zarr

from u_rpn.losses import bce


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

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)

    def test_equals_positives(self):
        pred = tf.constant(np.ones((10, 10)))
        target = tf.constant(np.ones((10, 10)))

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)

    def test_equals_multiple(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        pred[0:2, 0:2] = 1
        target[0:2, 0:2] = 1

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)

    def test_diff_small_class(self):
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        target[0:2, 0:2] = 0

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 7.71, places=2)

    def test_diff_big_class(self):
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))

        target[0:2, 0:2] = 1

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 7.71, places=2)

    def test_diff_both_class(self):
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))

        target[0:2, 0:2] = 1
        pred[0:2, 0:2] = 0

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertGreater(bce.WeightedBCE()(target, pred).numpy(), 15)


class TestTernaryBCE(TestCase):
    """ Test suite for ternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with known problematic inputs.
    """

    def test_all_positives(self):
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCE()(target, pred).numpy(), 0)

    def test_all_negatives(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCE()(target, pred).numpy(), 0)

    def test_known_error(self):
        target = zarr.load('../in/gt.zarr')
        pred = zarr.load('../in/masks.zarr')

        bce.WeightedTernaryBCE()(target, pred)


class TestTernaryReverseBCE(TestCase):
    """ Test suite for ternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with known problematic inputs.
    """

    def test_all_positives(self):
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCEReverse()(target, pred).numpy(), 0)

    def test_all_negatives(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCEReverse()(target, pred).numpy(), 0)

    def test_known_error(self):
        target = zarr.load('../in/gt.zarr')
        pred = zarr.load('../in/masks.zarr')

        bce.WeightedTernaryBCEReverse()(target, pred)


class TestQuaternaryBCE(TestCase):
    """ Test suite for quaternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with known problematic inputs.
    """

    def test_all_positives(self):
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        self.assertAlmostEqual(bce.WeightedQuaternaryBCE()(target, pred).numpy(), 0)

    def test_all_negatives(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        self.assertAlmostEqual(bce.WeightedQuaternaryBCE()(target, pred).numpy(), 0)

    def test_known_error(self):
        target = zarr.load('../in/gt.zarr')
        pred = zarr.load('../in/masks.zarr')

        bce.WeightedQuaternaryBCE()(target, pred)
