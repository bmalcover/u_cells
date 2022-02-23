""" This module generates an augmentation of the dataset.

The augmentation is done with the imaug library. The parameters of this library, defined in this
module are the same than the ones used in the python notebook.

Copyright (C) 2020-2022  Miquel Miró Nicolau, UIB
Written by Miquel Miró (UIB), 2021
"""

import os
import json
import warnings

import numpy as np
import cv2

import imgaug.augmenters as iaa
import imgaug as ia
import imgaug

import normalize_dataset

warnings.filterwarnings("ignore")

MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad",
                   "Affine", "PiecewiseAffine"]


def hook(images, augmenter, parents, default):
    """Determines which augmenters to apply to masks."""
    return augmenter.__class__.__name__ in MASK_AUGMENTERS


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

augmentation = [  # apply the following augmenters to most images
    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
    iaa.Flipud(0.2),  # vertically flip 20% of all images

    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45),  # rotate by -45 to +45 degrees
        shear=(-16, 16),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=0,  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL
        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
    # execute 0 to 5 of the following (less important) augmenters per image
    # don't execute all of them, as that would often be way too strong
    iaa.SomeOf((0, 5),
               [
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                       iaa.AverageBlur(k=(2, 7)),
                       # blur image using local means with kernel sizes between 2 and 7
                       iaa.MedianBlur(k=(3, 11)),
                       # blur image using local medians with kernel sizes between 2 and 7
                   ]),
               ],
               random_order=True)]
augmentation = iaa.Sequential(augmentation)

TO_GENERATE = 300
OUTPUT_FOLDER = os.path.join("..", "..", "out", "augmented")


def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(np.array([[c] for c in contour]))  # Converts points to contours
    return ((origin[0] // tolerance_factor) * tolerance_factor) * cols + origin[1]


def main():
    path_regions = os.path.join("..", "..", "in", "train")
    images_info_path = os.path.join(path_regions, "via_region_data.json")
    # images_info = list(images_info.values())  # We do not need the keys
    images_info = normalize_dataset.get_raw_img_and_info(path_regions, images_info_path)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    generation = 0
    bboxes_json = {}
    while generation < TO_GENERATE:
        img_idx = generation % len(images_info)
        img_id, img, regions, mask, cells_type = images_info[img_idx]

        # Store shapes before augmentation to compare
        img_shape = img.shape
        mask_shape = mask.shape

        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        img = det.augment_image(img)

        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))

        # Verify that shapes didn't change
        assert img.shape == img_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

        img, improved_mask, regions_augmented = normalize_dataset.normalize_img_mask(img, mask)

        bboxes_json[generation] = {'regions': regions_augmented, 'cell_class': cells_type}
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{generation}.jpg"), img)

        improved_mask = np.dstack(improved_mask)

        assert improved_mask.shape[-1] == len(regions_augmented)

        np.save(os.path.join(OUTPUT_FOLDER, f"{generation}.npy"), improved_mask)
        print(f"Generation {generation}")

        generation += 1

    with open(os.path.join(OUTPUT_FOLDER, 'bboxes.json'), 'w+') as f:
        json.dump(bboxes_json, f)


main()
