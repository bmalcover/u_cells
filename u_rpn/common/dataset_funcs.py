""" Module containing common functions for the main programs.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""
import json
import os
from typing import Tuple

import cv2
import numpy as np
import skimage
import skimage.color
import skimage.draw
import skimage.io
import skimage.transform

from u_rpn.common import utils


def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(
        np.array([[c] for c in contour])
    )  # Converts points to contours
    return ((origin[0] // tolerance_factor) * tolerance_factor) * cols + origin[1]


def get_raw_img_and_info(info_path: str):
    """Returns the raw image and the information of the image.

    Args:
        info_path: String with the path to the information file.

    Yields:
        Image id, image, regions_list, mask, cell type.
    """
    path, _ = os.path.split(info_path)

    images_info = json.load(open(info_path))
    images_info = list(images_info.values())  # We do not need the keys

    for img_idx, img_info in enumerate(images_info):
        filename = img_info["filename"]

        img_path = os.path.join(path, filename)
        image = cv2.imread(img_path)

        regions = list(img_info["regions"].values())
        mask = np.zeros([image.shape[0], image.shape[1], len(regions)], dtype=np.uint8)
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

        yield img_idx, image, regions_list, mask, cell_type


def get_normalized_data(path: str, extension: str = "jpg"):
    """Yields the image and mask data from a normalized dataset.

    Args:
        path:
        extension:

    Yields:
        Image id, image, regions_list, mask, cell type.
    """
    dataset_dir, _ = os.path.split(path)
    image_data = json.load(open(path))

    for img_idx, info in image_data.items():
        regions_list = info["regions_list"]
        regions_list = [np.array(r) for r in regions_list]

        cells_class = info["cell_class"]

        image_path = os.path.join(dataset_dir, f"{img_idx}.{extension}")
        mask_path = os.path.join(dataset_dir, f"{img_idx}.npy")

        image = skimage.io.imread(image_path)
        mask = np.load(mask_path)

        yield img_idx, image, regions_list, mask, cells_class


def normalize_img_mask(
    img: np.array, mask: np.array, min_dim: int = 512, max_dim: int = 512
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Normalize and resize the image and mask.

    Args:
        img: Numpy array of the image to be normalized.
        mask: Numpy array of the mask to be normalized.
        min_dim: Integer with the minimum dimension of the image.
        max_dim: Integer with the maximum dimension of the image.

    Returns:
        img: Normalized image.
        mask: Normalized mask.
    """

    img, window, scale, padding, crop = utils.resize_image(
        img, min_dim=min_dim, min_scale=0, max_dim=max_dim, mode="square"
    )
    mask = utils.resize_mask(mask, scale, padding, crop)
    regions_augmented = []
    improved_mask = []
    for idx_channel in range(mask.shape[-1]):
        contours, _ = cv2.findContours(
            mask[:, :, idx_channel], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            max_contour = contours[0]

            regions_augmented.append(
                [[int(c[0][0]), int(c[0][1])] for c in max_contour]
            )
    regions_augmented = sorted(
        regions_augmented, key=lambda x: get_contour_precedence(x, img.shape[1])
    )
    for r in regions_augmented:
        aux_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        cv2.drawContours(aux_mask, [np.array(r)], -1, 1, -1)

        improved_mask.append(aux_mask)

    return img, np.dstack(improved_mask), regions_augmented
