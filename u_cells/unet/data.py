# -*- coding: utf-8 -*-
from typing import Tuple, Union
from enum import Enum
import os

import numpy as np
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils as KU

CODES = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]


class DecodeMode(Enum):
    CELLS = 1
    CELLS_BCK = 2
    MULTICHANEL_N_BCK = 3
    MULTICHANEL_BCK = 4


def decode(gt_img: np.ndarray, mode: DecodeMode):
    """ Decodes the input image into a mask for class or a background image.
    
    To be able to handle multiple channels images (more than three), instead of saving corrupt img
    files with more than three channels we define a unique code for any of the desirable channels.
      
    Args:
        gt_img (numpy array): Ground truth image of 3 channels. The image is encoded to contain
                              more than one channel
        mode (DecodeMode): Mode to decode

    Returns:

    """
    channels = []
    if mode in (DecodeMode.MULTICHANEL_N_BCK, DecodeMode.MULTICHANEL_BCK):
        for c in CODES:
            channel = np.argmax(c)
            mask_aux = gt_img[:, :, channel] == c[channel]
            mask_aux = mask_aux.astype(np.uint8) * 255

            channels.append(mask_aux)

        decoded_img = np.dstack(channels)
    else:
        decoded_img = (gt_img[:, :, 0] < 240) & (gt_img[:, :, 1] < 240) & (gt_img[:, :, 2] < 240)
        decoded_img = decoded_img.astype(np.uint8) * 255

        channels.append(decoded_img)

    if mode in (DecodeMode.MULTICHANEL_BCK, DecodeMode.CELLS_BCK):
        # We add the background if one of the decoded options with background is selected.
        bck_img = (gt_img[:, :, 0] > 240) & (gt_img[:, :, 1] > 240) & (gt_img[:, :, 2] > 240)
        bck_img = bck_img.astype(np.uint8) * 255

        channels.append(bck_img)
        decoded_img = np.dstack(channels)

    return decoded_img


class DataGenerator(KU.Sequence):

    def __init__(self, steps: int, batch_size: int, path: str, image_folder: str, mask_folder: str,
                 aug_dict: dict, dcd_mode: Union[DecodeMode, None], img_color_mode: str,
                 mask_color_mode: str, target_size: Tuple[int, int], seed: int = 1,
                 do_regression: Union[str, None] = None, normalize_region_data: bool = False):
        self.__steps = steps
        self.__path = path
        self.__decode_mode = dcd_mode

        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)

        self.__image_generator = image_datagen.flow_from_directory(
            path,
            classes=[image_folder],
            class_mode=None,
            color_mode=img_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed
        )

        self.__mask_generator = mask_datagen.flow_from_directory(
            path,
            classes=[mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed
        )

        self.__stadistic = []
        if do_regression is not None:
            self.__regression_data = self.__get_regions_info(do_regression, normalize_region_data)
        else:
            self.__regression_data = None

        self.__generator = self.__get_merged_info()

    def __get_regions_info(self, path: str, normalize: bool = False):
        """ Gets the information of the regions.

        The information is stored with VIA 2.0 format. The regions are saved as a dictionary for each image, the key is
        "regions". This key is a list of regions. We are only interested on the number of regions

        TODO: Add the ability to return also the region depending on a parameter

        Args:
            path (str): String containing the information of the regions.

        Returns:

        """
        info = json.load(open(path))

        mean = 0
        std = 1

        if normalize:
            mean_std = np.array([len(v["regions"]) for v in info.values()])
            mean = mean_std.mean()
            std = mean_std.std()

        self.__stadistic = [mean, std]

        info = {k: (len(v["regions"]) - mean) / std for k, v in info.items()}

        return info

    def __get_merged_info(self):
        """ Yields the information from the dataset.

        Returns:

        """
        for img, mask in zip(self.__image_generator, self.__mask_generator):
            img = img / 255

            if self.__decode_mode is not None:
                if self.__decode_mode is DecodeMode.CELLS:
                    new_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))

                    for m_idx in range(0, mask.shape[0]):
                        m = mask[m_idx, :, :]
                        new_mask[m_idx, :, :] = decode(m, self.__decode_mode)
                else:
                    new_mask = np.zeros(
                        (mask.shape[0], mask.shape[1], mask.shape[2], self.__decode_mode.value))

                    for m_idx in range(0, mask.shape[0]):
                        m = mask[m_idx, :, :, :]
                        new_mask[m_idx, :, :, :] = decode(m, self.__decode_mode)
                mask = new_mask

            mask = mask / 255

            if self.__regression_data is not None:
                idx = (self.__image_generator.batch_index - 1) * self.__image_generator.batch_size
                batch_filenames = self.__image_generator.filenames[idx: idx + self.__image_generator.batch_size]

                n_cells = []
                for filename in batch_filenames:
                    region_key = os.path.split(filename)[-1].split(".")[0]

                    n_cells.append(len(list(self.__regression_data)[int(region_key)]))

                n_cells = np.array(n_cells)

                yield img, {"img_out": mask, 'regressor_output': n_cells}

    @property
    def mean(self) -> Union[int, float]:
        assert len(self.__stadistic) == 2

        return self.__stadistic[0]

    @property
    def std(self) -> Union[int, float]:
        assert len(self.__stadistic) == 2

        return self.__stadistic[1]

    def __len__(self):
        return self.__steps

    def __getitem__(self, idx):
        return next(self.__generator)
