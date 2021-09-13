from enum import Enum
import random
import json
import glob
import os

import imgaug.augmenters as iaa
from skimage import transform
import numpy as np
import skimage
import tqdm
import cv2

CODES = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]


class DecodeMode(Enum):
    CELLS = 1
    CELLS_BCK = 2
    MULTICHANNEL_N_BCK = 3
    MULTICHANNEL_BCK = 4


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
    if mode in (DecodeMode.MULTICHANNEL_N_BCK, DecodeMode.MULTICHANNEL_BCK):
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

    if mode in (DecodeMode.MULTICHANNEL_BCK, DecodeMode.CELLS_BCK):
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

        regions = region_info[name].values()
        regions_lens = map(lambda x: len(x['shape_attributes']['all_points_x']), regions)

        min_points = []
        last_point = 0
        for idx_region, region_l in enumerate(regions_lens):
            points = points_aug[0][last_point:last_point + region_l, :].astype(int)
            dist_2_origin = list(map(lambda x: np.linalg.norm(x - np.array([0, 0])), points))

            min_points.append(min(dist_2_origin))

            if to_mask:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                mask = cv2.drawContours(mask, [points], -1, 1, -1)

                mask = cv2.resize(mask, output_shape)

                masks.append(mask)
            else:
                region_out[f"{idx}.png"]["regions"][str(idx_region)] = {
                    "shape_attributes": {'all_points_x': list(points[:, 0]),
                                         'all_points_y': list(points[:, 1])}}

            last_point += region_l

        idx_min_points = np.argsort(min_points)

        out_path = os.path.join(output_folder, str(idx).zfill(3) + ".png")
        cv2.imwrite(out_path, img_aug * 255)

        if to_mask:
            with open(os.path.join(output_folder, str(idx).zfill(3) + ".npy"), 'wb+') as f:
                masks = np.dstack(masks)
                masks = masks[:, :, idx_min_points]
                np.save(f, np.array(masks))

    if not to_mask:
        with open(os.path.join(output_folder, "regions.json"), "w") as outfile:
            json.dump(region_out, outfile)
