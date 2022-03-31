# -*- coding: utf-8 -*-
""" Test suite for reduced dataset and reduced data generator classes.

Written by: Miquel Miró Nicolau (UIB)
"""
from unittest import TestCase

import numpy as np

from u_rpn.configurations import cell_configuration as cell_config
from u_rpn.data import datasets
from u_rpn.data import rpn as u_rpn_data


class TestReducedDataset(TestCase):
    """ Test the ReducedDataset class.

    Cases:
        - Test the size of the input.
        - Test whether the first and second batch are different.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = cell_config.CellConfig()
        config.IMAGE_SHAPE = [512, 512, 3]

        dataset = datasets.ErithocytesPreDataset("../out/pre_calculate/train", "data.json",
                                                 divisor=255)
        dataset.prepare()

        generator = u_rpn_data.DataGenerator(50, dataset, pre_calculated=True, config=config,
                                             phantom_output=True, shuffle=False,
                                             size_anchors=dataset.anchors)

        self.__generator = generator

    def test_shape_data(self):
        for info, phantom in self.__generator:
            break

        self.assertAlmostEqual(info[0].shape[-1], 3)
        self.assertAlmostEqual(info[1].shape[-1], 81)
        self.assertAlmostEqual(info[2].shape[1], 49152)
        self.assertAlmostEqual(info[3].shape[1], 256)

    def test_difference_between_batches(self):
        images = []
        for i, (info, phantom) in enumerate(self.__generator):
            images.append(info[0])
            if i == 2:
                break

        n_zeros_f_img = np.count_nonzero(images[0] > 0.5)
        n_zeros_s_img = np.count_nonzero(images[1] > 0.5)

        self.assertNotEquals(n_zeros_s_img, n_zeros_f_img)

