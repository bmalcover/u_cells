# -*- coding: utf-8 -*-
""" Script for the detection of high density areas in a given image.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""
import json
import os

import cv2
import numpy as np

import common_main_funcs as cmf

DATA_FOLDER = os.path.join(".", "in", "train")
OUTPUT_FOLDER = os.path.join(".", "out", "hda")
WINDOWS_SIZE = (1024, 1024)
DENSITY_THRESH = 0.05


def get_sliding_window(img, mask, size):
    """ Yields a sliding window of the given size.
    """
    assert (img.shape[0] == mask.shape[0]) and (
            img.shape[1] == mask.shape[1]), "Image and mask must have the same shape."

    for y in range(0, img.shape[0] - size[0], size[0]):
        for x in range(0, img.shape[1] - size[1], size[1]):
            yield x, y, img[y:y + size[0], x:x + size[1], :], mask[y:y + size[0], x:x + size[1], :]


def region_inside_window(region, window):
    initial_coord, size = window

    normalized_region = region - initial_coord

    is_inside = np.any((normalized_region[:, 0] > 0) & (normalized_region[:, 1] > 0) &
                       (normalized_region[:, 0] < size[0]) & (normalized_region[:, 1] < size[1]))

    if not is_inside:
        normalized_region = None

    return is_inside, normalized_region


def main():
    """ Main function.
    """
    images_info_path = os.path.join(DATA_FOLDER, "via_region_data.json")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    bboxes_json = {}
    for img_idx, img, regions, mask, cell_type in cmf.get_raw_img_and_info(DATA_FOLDER,
                                                                           images_info_path):
        print(f"Start to thread the img {img_idx} ...")
        for window_idx, (x, y, img_w, mask_w) in enumerate(
                get_sliding_window(img, mask, WINDOWS_SIZE)):
            mask_aux = np.copy(mask_w)
            mask_aux = np.sum(mask_aux, axis=-1)
            positives_px = np.count_nonzero(mask_aux)

            proportion = positives_px / mask_aux.size
            if proportion > DENSITY_THRESH:
                print(f"Window {window_idx} accepted with a proportion of {proportion}")
                idx_img_window = f"{str(img_idx).zfill(2)}_{str(window_idx).zfill(2)}"
                window_info = ((x, y), WINDOWS_SIZE)

                reg_info = list(map(lambda r: region_inside_window(r, window_info), regions))
                are_inside, _ = zip(*reg_info)
                are_inside = np.array(are_inside)

                img_w, mask_w, regions_norm = cmf.normalize_img_mask(img_w, mask_w)

                cell_type_aux = list(np.array(cell_type, dtype=int)[are_inside])
                cell_type_aux = [int(ct) for ct in cell_type_aux]

                bboxes_json[idx_img_window] = {'regions': regions_norm,
                                               'cell_class': cell_type_aux}

                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{idx_img_window}.jpg"), img_w)
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{idx_img_window}.png"),
                            np.sum(mask_w, axis=-1, dtype=np.uint8) * 255)
                np.save(os.path.join(OUTPUT_FOLDER, f"{idx_img_window}.npy"),
                        mask_w)
    with open(os.path.join(OUTPUT_FOLDER, 'bboxes.json'), 'w+') as f:
        json.dump(bboxes_json, f)


main()
