# -*- coding: utf-8 -*-
""" Data generators used by the U-Net neural network.

"""
from enum import Enum
import itertools
import random
import json
import os
import glob

import cv2
import tqdm
import skimage
import numpy as np
from skimage import transform
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


def generate_data(n_images: int, input_path: str, output_folder: str, augmentation,
                  region_point_path: str, to_mask: bool = False, output_shape=None):
    """ Generate data an saves it to disk.

    The data generation is a slow task. This functions aims to generate a defined set of images, and
    regions, to improve the performance of the train process.

    Args:
        n_images (int):  Number of images to generate.
        input_path (str): Path (glob format) containing the set of files.
        output_folder (str): Path to output files.
        augmentation:
        region_point_path:
        to_mask:
        output_shape:

    """
    filenames = glob.glob(input_path)

    region_info = json.load(open(region_point_path))
    region_info = {k: v["regions"] for k, v in region_info.items()}

    region_out = {}

    augmentation = iaa.Sequential(augmentation)
    os.makedirs(output_folder, exist_ok=True)

    for idx in tqdm.tqdm(range(n_images)):
        if to_mask:
            masks = []
        else:
            region_out[str(idx) + ".png"] = {"regions": {}}

        filename = random.choice(filenames)
        _, name = os.path.split(filename)

        img = cv2.imread(filename)

        h_points = []
        v_points = []

        for region in region_info[name].values():
            region = region["shape_attributes"]

            h_points += region['all_points_x']
            v_points += region['all_points_y']

        points = np.column_stack((h_points, v_points))

        img_aug, points_aug = augmentation(images=[img], keypoints=[points])
        img_aug = img_aug[0]

        img_aug = skimage.transform.resize(img_aug, (output_shape[0], output_shape[1], 3))

        last_point = 0
        for idx_region, region in enumerate(region_info[name].values()):
            region = region["shape_attributes"]
            points = points_aug[0][last_point:last_point + len(region['all_points_x']), :].astype(
                int)

            if to_mask:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                mask = cv2.drawContours(mask, [points], -1, 1, -1)

                mask = cv2.resize(mask, output_shape)

                masks.append(mask)
            else:
                region_out[f"{idx}.png"]["regions"][str(idx_region)] = {
                    "shape_attributes": {'all_points_x': list(points[:, 0]),
                                         'all_points_y': list(points[:, 1])}}

            last_point += len(region['all_points_x'])

        out_path = os.path.join(output_folder, str(idx) + ".png")
        cv2.imwrite(out_path, img_aug * 255)

        if to_mask:
            with open(os.path.join(output_folder, str(idx) + ".npy"), 'wb+') as f:
                masks = np.dstack(masks)
                np.save(f, np.array(masks))

    if not to_mask:
        with open(os.path.join(output_folder, "regions.json"), "w") as outfile:
            json.dump(region_out, outfile)


class DataGenerator(KU.Sequence):

    def __init__(self, batch_size: int, steps: int, path: str, shape, output_size: int,
                 multi_type: bool = False, regression: bool = False, region_path: str = None,
                 rgb_input: bool = True, augmentation=None, load_from_cache: bool = False,
                 do_background: bool = False, unified_batch: bool = False):
        if "*" not in path and region_path is None:
            raise Exception("Regions path or a path with global format needed")

        self.__base_path = path
        self.__steps = steps
        self.__multi_type = multi_type
        self.__regression = regression
        self.__shape = shape
        self.__rgb = rgb_input
        self.__output_size = output_size
        self.__batch_size = batch_size

        self.__load_from_cache = load_from_cache
        self.__augmentation = iaa.Sequential(augmentation)
        self.__do_background = do_background
        self.__unified_batch = unified_batch

        if region_path is not None:
            self.__region_data = self.__get_regions_info(region_path)
            self.__keys = list(self.__region_data.keys())
        else:
            self.__keys = None

    def __get_regions_info(self, path: str) -> dict:
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

    def __len__(self):
        """ Return number of batches

        """
        return self.__steps

    def __load_cache(self, path: str, filename: str, n_channels: int):
        """ Loads cache masks.

        Args:
            path:
            filename:
            n_channels:

        Returns:

        """
        mask_path, _ = os.path.split(path)
        _, name_path = os.path.split(filename)

        mask = None
        with open(os.path.join(mask_path, f"{name_path.split('.')[0]}.npy"),
                  'rb') as f:
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

        if mask is None:
            mask = np.zeros((self.__shape[0], self.__shape[1], n_channels))

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
            """ Creates a mask from a set of points

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
            c_mask = cv2.resize(c_mask, shape)

            return c_mask

        mask = np.zeros(out_shape, dtype=np.float32)

        # We pass the points into numpy array format
        h_points = [r["shape_attributes"]['all_points_x'] for r in regions.values()]
        v_points = [r["shape_attributes"]['all_points_y'] for r in regions.values()]
        regions_size = list(map(lambda x: len(x), h_points))

        h_points = itertools.chain.from_iterable(h_points)
        v_points = itertools.chain.from_iterable(v_points)

        points = np.column_stack((h_points, v_points))

        if augmentation is not None:
            img_aug, points_aug = augmentation(images=[image], keypoints=[points])
            input_img, points = img_aug[0], points_aug[0]

        n_regions_points = 0
        for idx_channel, (n_points) in enumerate(list(regions_size)):
            if idx_channel == out_shape[-1]:
                break

            region_points = points[n_regions_points:n_regions_points + n_points]

            n_regions_points = n_regions_points + n_points

            channel_mask = draw_polygon(region_points, (image.shape[0], image.shape[1]))

            mask[:, :, idx_channel] = channel_mask
            idx_channel += 1

        return mask, len(regions)

    def __getitem__(self, idx):
        """ Returns a batch to train.

        Args:
            idx:

        Returns:

        """
        input_batch = []
        masks = []
        regressors = []

        output_shape = (self.__shape[0], self.__shape[1], self.__output_size)

        files = self.__keys

        if files is None:
            files = glob.glob(self.__base_path)

        for n_batch in range(0, self.__batch_size):
            idx = (idx + n_batch) % len(files)
            filename = files[idx]

            input_shape = self.__shape
            if self.__rgb:
                input_shape = (self.__shape[0], self.__shape[1], 3)

            path = filename
            if not self.__load_from_cache:
                path = os.path.join(self.__base_path, path)

            input_img = cv2.imread(path)
            input_img = skimage.transform.resize(input_img, input_shape)

            if self.__load_from_cache:
                mask, n_regions = self.__load_cache(path, filename, self.__output_size)

            else:
                mask, n_regions = DataGenerator.__draw_regions(self.__region_data[filename],
                                                               input_img, output_shape,
                                                               self.__augmentation)

            output_size = self.__output_size
            if self.__do_background:
                foreground = np.sum(mask, axis=-1)  # We merge all the channels with info
                background = np.zeros_like(foreground)

                background[foreground == 0] = 1
                mask = np.dstack([background, mask])

                output_size += 1

            if not self.__multi_type:
                mask = np.sum(mask, axis=-1)

            mask = mask.reshape((self.__shape[0], self.__shape[1], output_size))

            masks.append(mask)
            regressors.append(n_regions)

            input_batch.append(input_img)

        if self.__unified_batch:
            masks = [list(np.swapaxes(np.swapaxes(m, 0, 2), 1, 2)) for m in masks]

            masks = list(itertools.chain.from_iterable(masks))
            masks = np.dstack(masks)
        else:
            masks = np.array(masks)

        input_batch = np.array(input_batch)
        regressors = np.array(regressors)

        if self.__regression:
            return input_batch, {"img_out": masks, 'regressor_output': regressors}
        else:
            return input_batch, {"img_out": masks}
