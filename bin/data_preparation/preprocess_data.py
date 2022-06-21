# -*- coding: utf-8 -*-
""" This module normalize the dataset to be able to use it with the generators.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""

import json
import os
import warnings

import numpy as np
import tqdm
import zarr

from u_rpn.configurations import cell_configuration as cell_config
from u_rpn.data import datasets as rpn_datasets
from u_rpn.data import rpn as rpn_data

warnings.filterwarnings("ignore")

DATA_FOLDER = os.path.join("./", "out", "normalized")
SAVE_WHOLE_BATCH = True

if SAVE_WHOLE_BATCH:
    OUTPUT_FOLDER = os.path.join("./", "out", "pre_calculate_m", "train")
else:
    OUTPUT_FOLDER = os.path.join("./", "out", "pre_calculate", "train")

INPUT_SIZE = [512, 512, 3]
STEP_IMPROVE = 50


def main() -> None:
    config = cell_config.CellConfig()
    config.IMAGE_SHAPE = INPUT_SIZE

    if not SAVE_WHOLE_BATCH:
        config.BATCH_SIZE = 1

    dataset_train = rpn_datasets.ErithocytesDataset(
        [("cell", 1, "cell")], "bboxes.json"
    )
    dataset_train.load_cell(os.path.join("in", "erit"), rpn_datasets.Subset.TRAIN)
    dataset_train.prepare()

    train_generator = rpn_data.DataGenerator(
        300, dataset_train, config, shuffle=False, phantom_output=True
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    is_info_saved = False
    for idx, ((images, gt_masks, rpn_match, rpn_bbox, gt_class_ids), _) in tqdm.tqdm(
        enumerate(train_generator), desc="Generating processed data"
    ):
        if not is_info_saved:
            with open(os.path.join(OUTPUT_FOLDER, "data.json"), "w+") as f:
                json.dump(
                    {
                        "total_matches": rpn_match.shape[1],
                        "whole_batch": int(SAVE_WHOLE_BATCH),
                    },
                    f,
                )
            is_info_saved = True
        s_idx = str(idx).zfill(3)

        folder = os.path.join(OUTPUT_FOLDER, s_idx)
        os.makedirs(folder, exist_ok=True)

        if SAVE_WHOLE_BATCH:
            zarr.save(os.path.join(folder, "image.zarr"), images * 255)
            zarr.save(os.path.join(folder, "mask.zarr"), gt_masks)
            zarr.save(os.path.join(folder, "matches.zarr"), rpn_match)
            np.save(os.path.join(folder, "bboxes.npy"), rpn_bbox)
            np.save(os.path.join(folder, "gt_class.npy"), gt_class_ids)
        else:
            zarr.save(os.path.join(folder, "image.zarr"), images[0] * 255)
            zarr.save(os.path.join(folder, "mask.zarr"), gt_masks[0])
            zarr.save(os.path.join(folder, "matches.zarr"), rpn_match[0])
            np.save(os.path.join(folder, "bboxes.npy"), rpn_bbox[0])
            np.save(os.path.join(folder, "gt_class.npy"), gt_class_ids[0])


if __name__ == "__main__":
    main()
