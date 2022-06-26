""" Data generators used by the U-Net neural network.

"""
import glob
import itertools
import json
import os
from enum import Enum

import cv2
import imgaug.augmenters as iaa
import numpy as np
import skimage
from skimage import transform  # noqa: F401
from tensorflow.keras import utils as KU


class DataFormat(Enum):
    CACHE = 1
    MULTI_MASK = 2
    MASK = 3


class DataGenerator(KU.Sequence):
    def __init__(
        self,
        batch_size: int,
        steps: int,
        path: str,
        shape,
        output_size: int,
        data_format: DataFormat,
        mask_path: str = None,
        region_path: str = None,
        augmentation=None,
        background: bool = False,
        foreground: bool = False,
        rgb: bool = False,
    ):
        if "*" not in path and region_path is None:
            raise Exception("Regions path or a path with global format needed")

        self.__steps = steps
        self.__shape = shape
        self.__output_size = output_size
        self.__batch_size = batch_size

        if data_format is None:
            data_format = DataFormat.MASK

        self.__data_format = data_format
        self.__augmentation = iaa.Sequential(augmentation)
        self.__background = background
        self.__foreground = foreground

        self.__rgb = rgb

        if mask_path is None:
            self.__masks = None
        else:
            self.__masks = sorted(glob.glob(mask_path))

        if region_path is not None:
            self.__regions = self.__get_regions_info(os.path.join(path, region_path))
            # We concatenate the base path to the file names
            self.__files = sorted(
                list(map(lambda x: os.path.join(path, x), self.__keys()))
            )
        else:
            self.__files = sorted(glob.glob(path))

    def __keys(self):
        return list(self.__regions.keys())

    @staticmethod
    def __get_regions_info(path: str) -> dict:
        """Gets the information of the regions.

        The information is stored with VIA 2.0 format. The regions are saved as a dictionary for
        each image, the key is "regions". This key is a list of regions. On the other hand we also
        are interest in the type of each region, this type is defined as an integer.

        Args:
            path (str): String containing the information of the regions.

        Returns:

        """
        info = json.load(open(path))

        info = {k: v["regions"] for k, v in info.items()}

        return info

    def __len__(self):
        """Return number of batches"""
        return self.__steps

    def __load_cache(self, path_img: str, n_channels: int):
        """Loads cache masks.

        Args:
            path_img (str):
            n_channels:

        Returns:

        """
        base_path, name_path = os.path.split(path_img)

        mask = np.zeros((self.__shape[0], self.__shape[1], n_channels))
        with open(os.path.join(base_path, f"{name_path.split('.')[0]}.npy"), "rb") as f:
            mask = np.load(f)
            # Mask has a shape of (shape[0], shape[1], number of channels with objects)
            mask = mask.reshape((mask.shape[0], mask.shape[1], -1))

            n_regions = mask.shape[-1]

            if mask.shape[-1] < self.__output_size:
                diff = self.__output_size - mask.shape[-1]

                # Depth == difference between mask and output size
                aux_mask = np.zeros((self.__shape[0], self.__shape[1], diff))
                mask = np.dstack((mask, aux_mask))
            elif mask.shape[-1] > self.__output_size:
                mask = mask[:, :, :n_channels]

        return mask, n_regions

    @staticmethod
    def __draw_regions(regions, image, out_shape, augmentation):
        """

        Args:
            regions:
            image:
            out_shape:
            augmentation:

        Returns:

        """

        def draw_polygon(pts, shape) -> np.ndarray:
            """Creates a mask from a set of points

            Draws the contours defined for the points passed as parameter. The list of points
            indicates the contour of and object.

            Args:
                pts: List of points
                shape: Tuple of two size

            Returns:

            """
            pts = pts.astype(int)
            c_mask = np.zeros(shape, dtype=np.uint8)

            cv2.drawContours(c_mask, [pts], -1, 1, -1)

            return c_mask

        mask = np.zeros(out_shape, dtype=np.float32)

        # We pass the points into numpy array format
        h_points = [r["shape_attributes"]["all_points_x"] for r in regions.values()]
        v_points = [r["shape_attributes"]["all_points_y"] for r in regions.values()]
        regions_size = list(map(len, h_points))

        h_points = list(itertools.chain.from_iterable(h_points))
        v_points = list(itertools.chain.from_iterable(v_points))

        points = np.column_stack((h_points, v_points))

        if augmentation is not None:
            img_aug, points_aug = augmentation(images=[image], keypoints=[points])
            image, points = img_aug[0], points_aug[0]

        n_regions_points = 0
        for idx_channel, (n_points) in enumerate(list(regions_size)):
            if idx_channel == out_shape[-1]:
                break

            region_points = points[n_regions_points : n_regions_points + n_points]

            n_regions_points = n_regions_points + n_points

            channel_mask = draw_polygon(region_points, (image.shape[0], image.shape[1]))
            channel_mask = cv2.resize(channel_mask, (out_shape[0], out_shape[1]))

            mask[:, :, idx_channel] = channel_mask
            idx_channel += 1

        return mask, image, len(regions)

    def __getitem__(self, idx: int):
        """Returns a batch to train.

        Args:
            idx (int):

        Returns:

        """
        input_batch = []
        masks = []

        output_shape = (self.__shape[0], self.__shape[1], self.__output_size)

        for n_batch in range(0, self.__batch_size):
            idx = (idx + n_batch) % len(self.__files)
            path = self.__files[idx]

            input_img = cv2.imread(path, int(self.__rgb))

            if self.__data_format is DataFormat.CACHE:
                mask, _ = self.__load_cache(path, self.__output_size)
            elif self.__data_format is DataFormat.MULTI_MASK:
                regions = self.__regions
                mask, input_img, _ = DataGenerator.__draw_regions(
                    regions[self.__keys()[idx]],
                    input_img,
                    output_shape,
                    self.__augmentation,
                )
            else:  # MASK:
                mask = cv2.imread(self.__masks[idx], -1)
                mask = cv2.resize(mask, self.__shape)

                mask = np.expand_dims(mask, 2)

            input_img = skimage.transform.resize(input_img, self.__shape)

            if self.__background or self.__foreground:
                background, foreground = self.__get_bg_fg(mask)

                if self.__foreground:
                    mask = foreground

                if self.__background:
                    mask = np.dstack([background, mask])

            masks.append(mask.reshape((self.__shape[0], self.__shape[1], -1)))
            input_batch.append(
                input_img.reshape((self.__shape[0], self.__shape[1], -1))
            )

        return np.array(input_batch), {"img_out": np.array(masks)}

    @staticmethod
    def __get_bg_fg(mask):
        """Extracts background and foreground of a mask.

        Args:
            mask:

        Returns:

        """
        foreground = np.sum(mask, axis=-1)  # We merge all the channels with info
        foreground[foreground > 1] = 1

        background = np.zeros_like(foreground)
        background[foreground == 0] = 1

        return background, foreground
