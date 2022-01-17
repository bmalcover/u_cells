# -*- coding: utf-8 -*-
""" This module normalize the dataset to be able to use it with the generators.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""

import os
import json
import warnings

import numpy as np
import cv2
import skimage
import skimage.io
import skimage.color
import skimage.transform
import skimage.draw

from u_cells.common import utils

warnings.filterwarnings("ignore")

OUTPUT_FOLDER = os.path.join(".", "out", "normalized")


def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(np.array([[c] for c in contour]))  # Converts points to contours
    return ((origin[0] // tolerance_factor) * tolerance_factor) * cols + origin[1]


def main():
    path_regions = os.path.join(".", "in", "train")
    images_info = json.load(open(os.path.join(path_regions, "via_region_data.json")))
    images_info = list(images_info.values())  # We do not need the keys

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # generation = 0
    bboxes_json = {}
    # while generation < TO_GENERATE:
    for img_idx, img_info in enumerate(images_info):
        # img_idx = generation % len(images_info)
        # img_info = images_info[img_idx]

        filename = img_info['filename']

        img_path = os.path.join(path_regions, filename)
        img = cv2.imread(img_path)

        regions = list(img_info['regions'].values())
        mask = np.zeros([img.shape[0], img.shape[1], len(regions)], dtype=np.uint8)
        cell_type = []

        for idx_reg, reg in enumerate(regions):
            cell_type.append(int(reg["type"] + 1))
            reg = reg["shape_attributes"]
            x_points = reg["all_points_x"]
            y_points = reg["all_points_y"]

            rr, cc = skimage.draw.polygon(y_points, x_points)
            mask[rr, cc, idx_reg] = 1

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

        img_idx = str(img_idx).zfill(2)
        bboxes_json[img_idx] = {'regions': regions_augmented, 'cell_class': cell_type}
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{img_idx}.jpg"), img)

        improved_mask = np.dstack(improved_mask)

        assert improved_mask.shape[-1] == len(regions_augmented), \
            "Mask and regions are not the same length"

        np.save(os.path.join(OUTPUT_FOLDER, f"{img_idx}.npy"), improved_mask)
        print(f"Image {img_idx}")

    with open(os.path.join(OUTPUT_FOLDER, 'bboxes.json'), 'w+') as f:
        json.dump(bboxes_json, f)


main()
