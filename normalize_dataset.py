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

import common_main_funcs as cmf

warnings.filterwarnings("ignore")

OUTPUT_FOLDER = os.path.join(".", "out", "normalized")


def main():
    path_regions = os.path.join(".", "in", "train")
    images_info_path = os.path.join(path_regions, "via_region_data.json")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    bboxes_json = {}

    for img_idx, img, regions, mask, cell_type in cmf.get_raw_img_and_info(path_regions,
                                                                           images_info_path):
        img, improved_mask, regions_augmented = cmf.normalize_img_mask(img, mask)

        img_idx = str(img_idx).zfill(2)
        bboxes_json[img_idx] = {'regions': regions_augmented, 'cell_class': cell_type}
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{img_idx.zfill(2)}.jpg"), img)

        assert improved_mask.shape[-1] == len(regions_augmented), \
            "Mask and regions are not the same length"

        np.save(os.path.join(OUTPUT_FOLDER, f"{img_idx.zfill(2)}.npy"), improved_mask)
        print(f"Image {img_idx.zfill(2)}")

    with open(os.path.join(OUTPUT_FOLDER, 'bboxes.json'), 'w+') as f:
        json.dump(bboxes_json, f)

# main()
