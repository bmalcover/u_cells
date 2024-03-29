"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
from abc import ABC
from typing import Optional

import numpy as np


class Config(ABC):
    """Base configuration class. For custom configurations, create a sub-class that inherits from
    this one and override properties that need to be changed.
    """

    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME: Optional[str] = None  # Override in sub-classes

    BATCH_SIZE = 4

    # Number of training steps per epoch. This doesn't need to match the size of the training set.
    # Tensorboard updates are saved at the end of each epoch, so setting this to a smaller number
    # means getting more frequent TensorBoard updates. Validation stats are also calculated at each
    # epoch end and they might take a while, so don't set this too small to avoid spending a lot of
    # time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    #   none:   No resizing or padding. Return the image unchanged.
    #   square: Resize and pad with zeros to get a square image
    #           of size [max_dim, max_dim].
    #   pad64:  Pads width and height with zeros to make them multiples of 64.
    #           If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #           up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #           The multiple of 64 is needed to ensure smooth scaling of feature
    #           maps up and down the 6 levels of the FPN pyramid (2**6=64).
    #   crop:   Picks random crops from the image. First, scales the image based
    #           on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #           size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #           IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes weights to explode. Likely due
    # to differences in optimizer implementation. The original library used 0.9, we used U-Net value
    # of 3e-5
    LEARNING_RATE = 3e-5

    # Use RPN ROIs or externally generated ROIs for training. Keep this True for most situations.
    # Set to False if you want to train the head branches on ROI generated by code rather than the
    # ROIs from the RPN. For example, to debug the classifier head without having to train the RPN.
    USE_RPN_ROIS = True

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    TOP_DOWN_PYRAMID_SIZE = 256
    # Threshold to accept predictions
    PRED_THRESHOLD = 0.8

    DO_MASK = True
    DO_MASK_CLASS = False
    DO_MERGE_BRANCH = False
    # Flag, if true the masks of the different objects are combined into a solo mask
    COMBINE_FG = False

    RAW_PREDICTION = True

    RANDOM_MASKS = False

    MAKE_BACKGROUND_MASK = False

    DYNAMIC_SIZE = False  # If true the size of the image is changed for each sample

    # Internal parameter
    __RPN_NUM_OUTPUTS = 4

    @property
    def RPN_NUM_OUTPUTS(self) -> int:
        num_outputs = self.__RPN_NUM_OUTPUTS

        if self.DO_MASK:
            num_outputs += 2  # If the mask is not in the output we removed two

        if self.DO_MASK_CLASS:
            num_outputs += 2

        if self.DO_MERGE_BRANCH:
            num_outputs += 2

        return num_outputs

    def __init__(self) -> None:
        """Set values of computed attributes."""

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT]
            )
        else:
            self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT]
            )

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def to_dict(self) -> dict:
        return {
            a: getattr(self, a)
            for a in sorted(dir(self))
            if not a.startswith("__") and not callable(getattr(self, a))
        }

    def __str__(self) -> str:
        """Display Configuration values."""
        res = "Configurations:\n"
        for key, val in self.to_dict().items():
            res += f"{key:30} {val} \n"
        res += "\n"

        return res
