""" Test suite for segmentations bce loss functions.

Written by: Miquel Miró Nicolau (UIB)
"""
from unittest import TestCase

import numpy as np
import tensorflow as tf

from u_rpn.losses import bce


class TestWeightedBCE(TestCase):
    """Test suite for weighted binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with half of the inputs equals to 1.
    """

    def test_all_negatives(self) -> None:
        pred = tf.constant(np.zeros((10, 10)))
        target = tf.constant(np.zeros((10, 10)))

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)

    def test_all_positives(self) -> None:
        pred = tf.constant(np.ones((10, 10)))
        target = tf.constant(np.ones((10, 10)))

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)

    def test_equals_multiple(self) -> None:
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        pred[0:2, 0:2] = 1
        target[0:2, 0:2] = 1

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)

    def test_diff_small_class(self) -> None:
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        target[0:2, 0:2] = 0

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 7.71, places=2)

    def test_diff_big_class(self) -> None:
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))

        target[0:2, 0:2] = 1

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 7.71, places=2)

    def test_diff_both_class(self) -> None:
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))

        target[0:2, 0:2] = 1
        pred[0:2, 0:2] = 0

        pred = tf.constant(pred)
        target = tf.constant(target)

        self.assertGreater(bce.WeightedBCE()(target, pred).numpy(), 15)

    def test_one_element_tensor(self) -> None:
        target = tf.constant(0, dtype=tf.float64)
        pred = tf.constant(0, dtype=tf.float64)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)


class TestTernaryBCE(TestCase):
    """Test suite for ternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with known problematic inputs.
    """

    def test_all_positives(self) -> None:
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCE()(target, pred).numpy(), 0)

    def test_all_negatives(self) -> None:
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCE()(target, pred).numpy(), 0)

    def test_one_element_tensor(self) -> None:
        target = tf.constant(0, dtype=tf.float64)
        pred = tf.constant(0, dtype=tf.float64)

        self.assertAlmostEqual(bce.WeightedBCE()(target, pred).numpy(), 0)


class TestTernaryReverseBCE(TestCase):
    """Test suite for ternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with known problematic inputs.
    """

    def test_all_positives(self) -> None:
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCEReverse()(target, pred).numpy(), 0)

    def test_all_negatives(self) -> None:
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        self.assertAlmostEqual(bce.WeightedTernaryBCEReverse()(target, pred).numpy(), 0)

    def test_one_element_tensor(self) -> None:
        target = tf.constant(0, dtype=tf.float64)
        pred = tf.constant(0, dtype=tf.float64)

        self.assertAlmostEqual(bce.WeightedTernaryBCEReverse()(target, pred).numpy(), 0)


class TestQuaternaryBCE(TestCase):
    """Test suite for quaternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with known problematic inputs.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.__loss = bce.WeightedQuaternaryBCE()

    def test_all_positives(self) -> None:
        pred = np.ones((10, 10))
        target = np.ones((10, 10))

        self.assertAlmostEqual(self.__loss(target, pred).numpy(), 0)

    def test_all_negatives(self) -> None:
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))

        self.assertAlmostEqual(self.__loss(target, pred).numpy(), 0)

    def test_one_element_tensor(self) -> None:
        target = tf.constant(0, dtype=tf.float64)
        pred = tf.constant(0, dtype=tf.float64)

        self.assertAlmostEqual(self.__loss(target, pred).numpy(), 0)


class TestUnsortedQuaternaryBCE(TestCase):
    """Suite of tests for unsorted quaternary binary cross entropy loss.

    Test cases:
        - Test with inputs all equals to 0.
        - Test with inputs all equals to 1.
        - Test with sorted inputs, with different values between the chanels but equals between GT
          and pred.
        - Test with unsorted inputs, with different values between the chanels but equals between GT

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__loss = bce.WU4BCE()

    def test_sorted_equal_ones(self) -> None:
        """Test with inputs all equals to 1.

        Returns:
            None.
        """
        pred = np.ones((1, 10, 10, 2))
        target = np.ones((1, 10, 10, 2))

        self.assertAlmostEqual(self.__loss(pred, target).numpy(), 0)

    def test_sorted_equal_zeros(self) -> None:
        """Test with inputs all equals to 0.

        Returns:
            None
        """
        pred = np.zeros((1, 10, 10, 2))
        target = np.zeros((1, 10, 10, 2))

        self.assertAlmostEqual(self.__loss(pred, target).numpy(), 0)

    def test_unsorted_equal(self) -> None:
        """Test with unsorted inputs, with different values between the chanels but equals between
        GT.

        Returns:
            None
        """
        pred = np.zeros((1, 10, 10, 2))
        target = np.zeros((1, 10, 10, 2))

        pred[:, 5:7, 5:7, 0] = 1
        target[:, 5:7, 5:7, 1] = 1

        self.assertLessEqual(self.__loss(pred, target).numpy(), 0.1)

    def test_big_unsorted_equal(self) -> None:
        """Test with unsorted inputs, with different values between the chanels but equals between
        GT and pred with a large number of channels.

        Returns:
            None
        """
        pred = np.zeros((1, 10, 10, 20))
        target = np.zeros((1, 10, 10, 20))

        pred[:, 5:7, 5:7, 0] = 1
        target[:, 5:7, 5:7, 10] = 1

        loss = self.__loss(pred, target).numpy()

        self.assertLessEqual(loss, 0.1)
