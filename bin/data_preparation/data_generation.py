# -*- coding: utf-8 -*-
""" Main script to test multiple library features

This script contains a small demo of the abilities of this library. In particular call to the
augmentation function and then use the generated data within a Keras generator.
"""

import imgaug as ia
import imgaug.augmenters as iaa

from u_rpn.data import unet as u_data


def main():
    def sometimes(aug: iaa.Affine) -> iaa.Sometimes:
        return iaa.Sometimes(0.5, aug)

    augmentation = [  # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        sometimes(
            iaa.Affine(
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
            )
        ),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(
            (0, 5),
            [
                iaa.OneOf(
                    [
                        iaa.GaussianBlur(
                            (0, 3.0)
                        ),  # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)),
                        # blur image using local medians with kernel sizes between 2 and 7
                    ]
                ),
            ],
            random_order=True,
        ),
    ]

    train_generator = u_data.DataGenerator(
        4,
        100,
        ".\\out_aug\\*.png",
        (512, 512),
        100,
        augmentation=augmentation,
        background=True,
        data_format=u_data.DataFormat.MASK,
    )

    for t, m in train_generator:
        print(type(t))
        break


if __name__ == "__main__":
    main()
