# -*- coding: utf-8 -*-
""" Script for the generation of plots for different loss functions.

Written by: Miquel Miró Nicolau (UIB)
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from u_rpn.common import config as rpn_config
from u_rpn.data import datasets as rpn_datasets
from u_rpn.data import rpn as rpn_data
from u_rpn.losses import segmentation

INPUT_SIZE = (512, 512, 81)
STEP_IMPROVE = 50


class CellConfig(rpn_config.Config):
    """Configuration for training on the toy  dataset.

    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cells"
    BATCH_SIZE = 6

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    BACKBONE_STRIDES = [4]
    RPN_ANCHOR_SCALES = [32]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + 3 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50
    #     LEARNING_RATE = 3e-01

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0
    PRED_THRESHOLD = 0.99999995

    IMAGE_SHAPE = [128, 128, 3]

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


config = CellConfig()
config.IMAGE_SHAPE = np.array(INPUT_SIZE)


def plot_loss(name_fn, loss_fn):
    random_input = np.random.rand(*INPUT_SIZE)
    random_input /= random_input.max()

    random_2_perf = np.copy(random_input)
    random_2_zero = np.copy(random_input)
    random_2_equals = np.copy(random_input)

    dataset_train = rpn_datasets.ErithocytesDataset([("cell", 1, "cell")], "bboxes.json")
    dataset_train.load_cell(os.path.join("in", "erit"), rpn_datasets.Subset.TRAIN)
    dataset_train.prepare()

    train_generator = rpn_data.DataGenerator(50, dataset_train, config, shuffle=False,
                                             phantom_output=True)

    info = None
    for info, _ in train_generator:
        break

    masks = info[1]
    mask_compressed = np.sum(masks[0, :, :, :], axis=-1)

    loss_perf = [loss_fn(masks[0, :, :, :], random_2_perf).numpy()]
    loss_zero = [loss_fn(masks[0, :, :, :], random_2_zero).numpy()]
    loss_equals = [loss_fn(masks[0, :, :, :], random_2_equals).numpy()]

    for i in tqdm(range(0, INPUT_SIZE[0] - STEP_IMPROVE, STEP_IMPROVE), desc="Horizontal axis"):
        for j in tqdm(range(0, INPUT_SIZE[1] - STEP_IMPROVE, STEP_IMPROVE), desc="Vertical axis"):
            for z in range(0, INPUT_SIZE[2] - STEP_IMPROVE, int(STEP_IMPROVE // 2)):
                random_2_perf[i: i + STEP_IMPROVE, j: j + STEP_IMPROVE, z: z + STEP_IMPROVE] = \
                    masks[0][i: i + STEP_IMPROVE, j: j + STEP_IMPROVE, z: z + STEP_IMPROVE]
                random_2_zero[i: i + STEP_IMPROVE, j: j + STEP_IMPROVE, z: z + STEP_IMPROVE] = 0
                if (z + 10) < masks.shape[-1]:
                    random_2_equals[i: i + STEP_IMPROVE, j: j + STEP_IMPROVE, z: z + STEP_IMPROVE] = \
                        mask_compressed[i: i + STEP_IMPROVE, j: j + STEP_IMPROVE]
                else:
                    random_2_equals[i: i + STEP_IMPROVE, j: j + STEP_IMPROVE,
                    z: z + STEP_IMPROVE] = 0
                loss_perf.append(loss_fn(masks[0, :, :, :], random_2_perf).numpy())
                loss_zero.append(loss_fn(masks[0, :, :, :], random_2_zero).numpy())
                loss_equals.append(loss_fn(masks[0, :, :, :], random_2_equals).numpy())

    px_moved = np.arange(len(loss_perf))
    plt.figure()
    plt.plot(px_moved, np.array(loss_perf), label="Perfecte")
    plt.plot(px_moved, np.array(loss_zero), label="Zero")
    plt.plot(px_moved, np.array(loss_equals), label="Equals")

    plt.legend()
    plt.title(name_fn)
    plt.show()


def main():
    functions = {"WBCE": segmentation.WeightedBCE(), "Dice RPN": segmentation.dice_rpn}

    for fun_name, func in functions.items():
        plot_loss(fun_name, func)


main()
