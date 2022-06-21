# -*- coding: utf-8 -*-
""" The datasets are python objects to read and load images for the RPN model.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2022
"""
import abc
import enum
import glob
import json
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import zarr


class Subset(enum.Enum):
    TRAIN = 1
    VALIDATION = 2

    def __str__(self):
        return str(self.name).lower()


def prepared_required(input_func):
    def aux(*args, **kwargs):
        if not args[0].prepared:
            raise EnvironmentError("First you must prepare the dataset object")
        return input_func(*args, **kwargs)

    return aux


class Dataset(ABC):
    """The base class for dataset classes.

    To use it, create a new class that adds functions specific to the dataset you want to use.
    For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.prepared = False
        self._whole_batch = False

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    @abstractmethod
    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function if you encounter images not in your
        dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i["source"] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)
        self.prepared = True

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info["source"] == source
        return info["id"]

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    @prepared_required
    def get_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array."""
        # Load image
        image = skimage.io.imread(self.image_info[image_id]["path"])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    @abc.abstractmethod
    def get_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with a binary mask per
                   instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(self, image_id):
        raise NotImplementedError

    @property
    def whole_batch(self):
        return self._whole_batch


class ErithocytesDataset(Dataset):
    """Dataset object to read and load images and masks for the RPN model of the erithocytes
    normalized dataset.

    """

    def __init__(
        self,
        dataset_classes: List[Tuple[str, int, str]],
        gt_file: str,
        extension: str = "jpg",
        divisor: Union[int, float] = 1,
        *args,
        **kwargs,
    ):
        self.__classes = dataset_classes
        self.__gt_file = gt_file
        self.__extension = extension
        self.__divisor = divisor

        super().__init__(*args, **kwargs)

    def load_cell(self, dataset_dir: str, subset: Subset):
        """Load a subset of the second erithocytes dataset.

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
            polygons = info["regions"]
            cells_class = (
                info["cell_class"]
                if len(self.__classes) > 1
                else np.ones(len(polygons))
            )

            image_path = os.path.join(dataset_dir, f"{image_key}.{self.__extension}")
            mask_path = os.path.join(dataset_dir, f"{image_key}.npy")

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cell",
                image_id=image_key,
                path=image_path,
                width=width,
                height=height,
                mask_path=mask_path,
                polygons=polygons,
                cells_class=cells_class,
            )

    @prepared_required
    def get_mask(self, image_id: int):
        """Generate instance masks for an image.

        Args:
            image_id: Integer, the image ID.

        Returns:
            masks:  A bool array of shape [height, width, instance count] with one mask per
                    instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        mask = np.load(info["mask_path"]) / self.__divisor

        return mask, info["cells_class"]

    @prepared_required
    def image_reference(self, image_id) -> str:
        """Return the path of the image.

        Args:
            image_id: Integer, the image ID.

        Returns:
            Unique id referring to a specific image.
        """
        info = self.image_info[image_id]
        return info["path"]

    def get_data(self, image_id):
        pass


class ErithocytesPreDataset(Dataset):
    """Dataset object to read and load images and masks for the RPN model of the erithocytes
    normalized dataset.

    """

    def get_mask(self, image_id):
        pass

    def image_reference(self, image_id):
        """Return the path of the image.

        Args:
            image_id: Integer, the image ID.

        Returns:
            Unique id referring to a specific image.
        """
        folder = os.path.join(self.__dataset_dir, image_id)

        return os.path.join(folder, "image.png")

    def __init__(
        self,
        dataset_dir,
        gt_file: str,
        extension: str = "jpg",
        divisor: Union[int, float] = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__dataset_dir = dataset_dir
        self.__gt_file = gt_file
        self.__extension = extension
        self.__divisor = divisor
        self.__size_anchors = 0
        self.__image_size = None
        self.__cache = {}

        self.prepare()

    @prepared_required
    def get_data(self, image_id):
        image_id = str(image_id).zfill(3)
        folder = os.path.join(self.__dataset_dir, image_id)

        masks = zarr.load(os.path.join(folder, "mask.zarr"))

        if image_id not in self.__cache:
            image = zarr.load(os.path.join(folder, "image.zarr")) / self.__divisor
            matches = zarr.load(os.path.join(folder, "matches.zarr"))
            bboxes = np.load(os.path.join(folder, "bboxes.npy"))
            gt_class_ids = np.load(os.path.join(folder, "gt_class.npy"))

            self.__cache[image_id] = (image, matches, bboxes, gt_class_ids)
        else:
            image, matches, bboxes, gt_class_ids = self.__cache[image_id]

        return image, masks, matches, bboxes, gt_class_ids

    @property
    def anchors(self):
        return self.__size_anchors

    def __len__(self):
        return self.__image_size

    def prepare(self, class_map=None):
        with open(os.path.join(self.__dataset_dir, self.__gt_file), "r") as f:
            aux = json.load(f)
            size_anchors = aux["total_matches"]
            whole_batch = bool(aux["whole_batch"])

            self._whole_batch = whole_batch
            self.__size_anchors = size_anchors
            self.__image_size = len(
                glob.glob(os.path.join(self.__dataset_dir, "**", "image.zarr"))
            )

            self.prepared = True
