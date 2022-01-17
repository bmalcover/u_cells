# -*- coding: utf-8 -*-
""" The datasets are python objects to read and load images for the RPN model.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""
import os
import enum
import json
from typing import List, Tuple

import skimage
import skimage.io
import skimage.color
import skimage.transform
import numpy as np

from . import rpn as rpn_data


class Subset(enum.Enum):
    TRAIN = 1
    VALIDATION = 2

    def __str__(self):
        return str(self.name).lower()


class ErithocytesDataset(rpn_data.Dataset):
    """ Dataset object to read and load images and masks for the RPN model of the erithocytes
    normalized dataset.

    """

    def __init__(self, dataset_classes: List[Tuple[str, int, str]], gt_file: str,
                 extension: str = "jpg", *args, **kwargs):
        self.__classes = dataset_classes
        self.__gt_file = gt_file
        self.__extension = extension

        super().__init__(*args, **kwargs)

    def load_cell(self, dataset_dir: str, subset: Subset):
        """ Load a subset of the second erithocytes dataset.

        Args:
            dataset_dir: String, root directory of the dataset.
            subset: Enumerate, Subset, indicating which subset to load: train or validation.
        """
        # Add classes. We have only one class to add.
        for class_source, class_id, class_name in self.__classes:
            self.add_class(class_source, class_id, class_name)

        # Train or validation dataset?
        dataset_dir = os.path.join(dataset_dir, str(subset))

        # Annotation following the format of VIA
        annotations = json.load(open(os.path.join(dataset_dir, self.__gt_file)))

        # Add images
        for image_key, info in annotations.items():
            polygons = info['regions']
            cells_class = info['cell_class'] if len(self.__classes) > 1 else np.ones(len(polygons))

            image_path = os.path.join(dataset_dir, f"{image_key}.{self.__extension}")
            mask_path = os.path.join(dataset_dir, f"{image_key}.npy")

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cell", image_id=image_key, path=image_path, width=width, height=height,
                mask_path=mask_path, polygons=polygons, cells_class=cells_class)

    def load_mask(self, image_id: int):
        """Generate instance masks for an image.

        Args:
            image_id: Integer, the image ID.

        Returns:
            masks:  A bool array of shape [height, width, instance count] with one mask per
                    instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        mask = np.load(info['mask_path'])

        return mask, info["cells_class"]

    def image_reference(self, image_id) -> str:
        """ Return the path of the image.

        Args:
            image_id: Integer, the image ID.

        Returns:
            Unique id referring to a specific image..
        """
        info = self.image_info[image_id]
        return info["path"]
