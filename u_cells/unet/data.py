# -*- coding: utf-8 -*-
from enum import Enum
import random
import json
import os
import glob

import skimage
from skimage import draw
from skimage import transform
import cv2
import numpy as np
import imgaug.augmenters as iaa

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


def generate_data(images_to_generate: int, input_path: str, output_folder: str, augmentation,
                  region_point_path: str):
    """ Generate data an saves it to disk.

    The data generation is a slow task. This functions aims to generate a defined set of images, and
    regions, to improve the performance of the train process.

    Args:
        images_to_generate:
        input_path:
        output_folder:
        augmentation:
        region_point_path:

    Returns:

    """
    filenames = glob.glob(input_path)

    region_info = json.load(open(region_point_path))
    region_info = {k: v["regions"] for k, v in region_info.items()}

    region_out = {}

    augmentation = iaa.Sequential(augmentation)

    for idx in range(images_to_generate):
        filename = random.choice(filenames)
        _, name = os.path.split(filename)

        img = cv2.imread(filename)
        points = region_info[name]

        img_aug, points_aug = augmentation(images=[img], keypoints=[points])

        out_path = os.path.join(output_folder, str(idx) + ".png")

        region_out[str(idx) + ".png"] = points_aug
        cv2.imwrite(out_path, img_aug)


class DataGenerator(KU.Sequence):

    def __init__(self, batch_size: int, steps: int, path: str, region_path: str, shape,
                 max_output: int, multi_type: bool = False, regression: bool = False,
                 rgb_input: bool = True, augmentation=None):
        self.__steps = steps
        self.__base_path = path
        self.__multi_type = multi_type
        self.__regression = regression
        self.__shape = shape
        self.__rgb = rgb_input
        self.__output_size = max_output
        self.__batch_size = batch_size

        self.__augmentation = iaa.Sequential(augmentation)

        self.__region_data = self.__get_regions_info(region_path)
        self.__keys = list(self.__region_data.keys())

    def __get_regions_info(self, path: str):
        """ Gets the information of the regions.

        The information is stored with VIA 2.0 format. The regions are saved as a dictionary for
        each image, the key is "regions". This key is a list of regions. On the other hand we also
        are interest in the type of each region, this type is defined as an integer.

        Args:
            path (str): String containing the information of the regions.

        Returns:

        """
        info = json.load(open(os.path.join(self.__base_path, path)))

        info = {k: v["regions"] for k, v in info.items()}

        return info

    def __draw_polygon(self, polygon, shape):
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0])
        channel_mask = np.zeros(shape)

        rr[rr >= shape[0]] = shape[0] - 1
        cc[cc >= shape[1]] = shape[1] - 1

        channel_mask[rr, cc] = 1
        channel_mask = cv2.resize(channel_mask, self.__shape)

        return channel_mask

    def __len__(self):
        return self.__steps

    def __getitem__(self, idx):
        """ Returns a batch to train.

        Args:
            idx:

        Returns:

        """
        input_batch = []
        masks = []
        regressors = []

        for n_batch in range(0, self.__batch_size):
            idx = (idx + n_batch) % len(self.__keys)

            filename = self.__keys[idx]

            input_img = os.path.join(self.__base_path, filename)
            input_img = cv2.imread(input_img)

            mask = np.zeros((self.__shape[0], self.__shape[1], self.__output_size),
                            dtype=np.float32)

            h_points = []
            v_points = []
            for region in self.__region_data[filename].values():
                region = region["shape_attributes"]

                h_points += region['all_points_x']
                v_points += region['all_points_y']

            points = np.column_stack((h_points, v_points))

            if self.__augmentation is not None:
                img_aug, points_aug = self.__augmentation(images=[input_img], keypoints=[points])
                input_img, points = img_aug[0], points_aug[0]

            n_regions_points = 0
            for idx_channel, (key, region) in enumerate(list(self.__region_data[filename].items())):
                if idx_channel == self.__output_size:
                    break

                region = region["shape_attributes"]
                region_points = points[
                                n_regions_points:n_regions_points + len(region['all_points_y'])]

                n_regions_points = n_regions_points + len(region['all_points_y'])

                channel_mask = self.__draw_polygon(region_points,
                                                   (input_img.shape[0], input_img.shape[1]))

                mask[:, :, idx_channel] = channel_mask
                idx_channel += 1

            mask = mask.reshape((self.__shape[0], self.__shape[1], self.__output_size))
            if self.__rgb:
                input_shape = (self.__shape[0], self.__shape[1], 3)
            else:
                input_shape = self.__shape

            input_img = skimage.transform.resize(input_img, input_shape).reshape(self.__shape[0],
                                                                                 self.__shape[1], 3)

            masks.append(mask)
            regressors.append(len(self.__region_data[filename].values()))

            input_batch.append(input_img)

        input_batch = np.array(input_batch)
        masks = np.array(masks)
        regressors = np.array(regressors)

        output = {"img_out": masks}
        if self.__regression:
            output['regressor_output'] = regressors

        return input_batch, output
