# -*- coding: utf-8 -*-
from enum import Enum
import json
import os

import skimage
from skimage import draw
from skimage import transform
import cv2
import numpy as np

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

    def __init__(self, steps: int, path: str, region_path: str, shape, max_output: int,
                 multi_type: bool = False, regression: bool = False, rgb_input: bool = True):
        self.__steps = steps
        self.__base_path = path
        self.__multi_type = multi_type
        self.__regression = regression
        self.__shape = shape
        self.__rgb = rgb_input
        self.__output_size = max_output
        # self.__decode_mode = dcd_mode

        # image_datagen = ImageDataGenerator(**aug_dict)
        # mask_datagen = ImageDataGenerator(**aug_dict)
        #
        # self.__image_generator = image_datagen.flow_from_directory(
        #     path,
        #     classes=[image_folder],
        #     class_mode=None,
        #     color_mode=img_color_mode,
        #     target_size=target_size,
        #     batch_size=batch_size,
        #     seed=seed
        # )
        #
        # self.__mask_generator = mask_datagen.flow_from_directory(
        #     path,
        #     classes=[mask_folder],
        #     class_mode=None,
        #     color_mode=mask_color_mode,
        #     target_size=target_size,
        #     batch_size=batch_size,
        #     seed=seed
        # )

        self.__stadistic = []
        self.__region_data = self.__get_regions_info(region_path)
        self.__keys = list(self.__region_data.keys())

        # if do_regression is not None:
        #     self.__regression_data = self.__get_regions_info(do_regression, normalize_region_data)
        # else:
        #     self.__regression_data = None
        #
        # self.__generator = self.__get_merged_info()

    def __get_regions_info(self, path: str):
        """ Gets the information of the regions.

        The information is stored with VIA 2.0 format. The regions are saved as a dictionary for
        each image, the key is "regions". This key is a list of regions. On the other hand we also
        are interest in the type of each region, this type is defined as an integer.

        TODO: Add the ability to return also the region depending on a parameter

        Args:
            path (str): String containing the information of the regions.

        Returns:

        """
        info = json.load(open(os.path.join(self.__base_path, path)))

        info = {k: v["regions"] for k, v in info.items()}

        return info

    # def __get_merged_info(self):
    #     """ Yields the information from the dataset.
    #
    #     Returns:
    #
    #     """
    #     for img, mask in zip(self.__image_generator, self.__mask_generator):
    #         img = img / 255
    #
    #         if self.__decode_mode is not None:
    #             if self.__decode_mode is DecodeMode.CELLS:
    #                 new_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
    #
    #                 for m_idx in range(0, mask.shape[0]):
    #                     m = mask[m_idx, :, :]
    #                     new_mask[m_idx, :, :] = decode(m, self.__decode_mode)
    #             else:
    #                 new_mask = np.zeros(
    #                     (mask.shape[0], mask.shape[1], mask.shape[2], self.__decode_mode.value))
    #
    #                 for m_idx in range(0, mask.shape[0]):
    #                     m = mask[m_idx, :, :, :]
    #                     new_mask[m_idx, :, :, :] = decode(m, self.__decode_mode)
    #             mask = new_mask
    #
    #         mask = mask / 255
    #         output = {"img_out": mask}
    #
    #         if self.__regression_data is not None:
    #             idx = (self.__image_generator.batch_index - 1) * self.__image_generator.batch_size
    #             batch_filenames = self.__image_generator.filenames[
    #                               idx: idx + self.__image_generator.batch_size]
    #
    #             n_cells = []
    #             for filename in batch_filenames:
    #                 region_key = os.path.split(filename)[-1].split(".")[0]
    #
    #                 n_cells.append(len(list(self.__regression_data)[int(region_key)]))
    #
    #             n_cells = np.array(n_cells)
    #
    #             output['regressor_output'] = n_cells
    #
    #         yield img, output

    # @property
    # def mean(self) -> Union[int, float]:
    #     assert len(self.__stadistic) == 2
    #
    #     return self.__stadistic[0]
    #
    # @property
    # def std(self) -> Union[int, float]:
    #     assert len(self.__stadistic) == 2
    #
    #     return self.__stadistic[1]

    def __len__(self):
        return self.__steps

    def __getitem__(self, idx):
        idx = idx % len(self.__keys)

        filename = self.__keys[idx]

        input_img = os.path.join(self.__base_path, filename)
        input_img = cv2.imread(input_img)

        mask = np.ones([input_img.shape[0], input_img.shape[1], self.__output_size], dtype=np.int32)

        for idx_channel, (key, region) in enumerate(self.__region_data[filename].items()):
            # If there are more regions than channels breaks the loop
            if idx_channel == self.__output_size:

                break

            region = region["shape_attributes"]
            rr, cc = skimage.draw.polygon(region['all_points_y'], region['all_points_x'])
            mask[rr, cc, idx_channel] = 1

        mask = mask.astype(np.float32)

        if self.__rgb:
            input_shape = (self.__shape[0], self.__shape[1], 3)
        else:
            input_shape = self.__shape

        input_img = skimage.transform.resize(input_img, input_shape)
        mask = skimage.transform.resize(mask, (self.__shape[0], self.__shape[1], mask.shape[2]))

        output = {"img_out": mask}

        if self.__regression:
            output['regressor_output'] = len(self.__region_data[filename].values())

        return input_img, mask
