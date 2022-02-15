# -*- coding: utf-8 -*-
""" Module containing common functions for the main programs.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""
import os
import json

import numpy as np
import cv2

import skimage
import skimage.io
import skimage.color
import skimage.transform
import skimage.draw

from u_cells.common import utils


def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(np.array([[c] for c in contour]))  # Converts points to contours
    return ((origin[0] // tolerance_factor) * tolerance_factor) * cols + origin[1]


def get_raw_img_and_info(path: str, info_path: str):
    images_info = json.load(open(info_path))
    images_info = list(images_info.values())  # We do not need the keys

    for img_idx, img_info in enumerate(images_info):
        filename = img_info['filename']

        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)

        regions = list(img_info['regions'].values())
        mask = np.zeros([img.shape[0], img.shape[1], len(regions)], dtype=np.uint8)
        cell_type = []

        regions_list = []
        for idx_reg, reg in enumerate(regions):
            cell_type.append(int(reg["type"] + 1))
            reg = reg["shape_attributes"]
            x_points = reg["all_points_x"]
            y_points = reg["all_points_y"]

            regions_list.append(np.transpose(np.vstack((x_points, y_points))))
            rr, cc = skimage.draw.polygon(y_points, x_points)
            mask[rr, cc, idx_reg] = 1

        yield img_idx, img, regions_list, mask, cell_type


def normalize_img_mask(img, mask):
    """ Normalize and resize the image and mask.


    Args:
        img: Numpy array of the image to be normalized.
        mask: Numpy array of the mask to be normalized.

    Returns:
        img: Normalized image.
        mask: Normalized mask.
    """

    img, window, scale, padding, crop = utils.resize_image(img, min_dim=400, min_scale=0,
                                                           max_dim=512, mode='square')
    mask = utils.resize_mask(mask, scale, padding, crop)
    regions_augmented = []
    improved_mask = []
    for idx_chann in range(mask.shape[-1]):
        contours, _ = cv2.findContours(mask[:, :, idx_chann], cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            max_contour = contours[0]

            regions_augmented.append([[int(c[0][0]), int(c[0][1])] for c in max_contour])
    regions_augmented = sorted(regions_augmented,
                               key=lambda x: get_contour_precedence(x, img.shape[1]))
    for r in regions_augmented:
        aux_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        cv2.drawContours(aux_mask, [np.array(r)], -1, 1, -1)

        improved_mask.append(aux_mask)

    return img, np.dstack(improved_mask), regions_augmented
