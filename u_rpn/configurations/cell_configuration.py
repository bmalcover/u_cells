""" Configuration for the eritocitos dataset.

Writen by: Miquel Miró Nicolau (UIB), 2022
"""
from u_rpn.common import config as rpn_config


class CellConfig(rpn_config.Config):
    # Give the configuration a recognizable name
    NAME = "cells"
    BATCH_SIZE = 6

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    BACKBONE_STRIDES = [4]
    RPN_ANCHOR_SCALES = [32]

    # Number of classes (including background)
    NUM_CLASSES = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50
    #     LEARNING_RATE = 3e-01

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0
    PRED_THRESHOLD = 0.99999995

    IMAGE_SHAPE: list = [128, 128, 3]

    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 400

    COMBINE_FG = False
    RANDOM_MASKS = False
    MAKE_BACKGROUND_MASK = False
    #     RPN_TRAIN_ANCHORS_PER_IMAGE = 200
    VALIDATION_STEPS = 10
    MAX_GT_INSTANCES = 81

    DO_MASK = True
    DO_MASK_CLASS = True
    DO_MERGE_BRANCH = False
    DYNAMIC_SIZE = True
