# -*- coding: utf-8 -*-
""" Metrics to measure the performance of segmentation methods.

"""
import numpy as np
from sklearn import metrics

from u_cells.common import utils


def relate_bbox_to_gt(bbox1, bbox2):
    """ Relates to sets of bounding boxes one to one with the IOU.

    Args:
        bbox1:
        bbox2:

    Returns:

    """
    overlaps = utils.compute_overlaps(bbox2, bbox1)
    relations = {}

    bbox2_id = 0
    while bbox2_id < len(bbox2) and len(relations.keys()) < len(bbox1):
        maximums = np.argsort(overlaps[bbox2_id])
        bbox1_id = 0

        while bbox1_id in maximums and bbox1_id < len(bbox1):
            bbox1_id += 1

        if bbox1_id < len(bbox1):
            relations[bbox1_id] = bbox2_id

        bbox2_id += 1

    return list(relations.keys()), list(relations.values())


def basic_metrics(ground, prediction):
    """ Calculate basic metrics from the GT and the prediction.

    Args:
        ground:
        prediction:

    Returns:

    """
    precision = metrics.precision_score(ground, prediction, average='weighted')
    recall = metrics.recall_score(ground, prediction, average='weighted')
    f1_score = metrics.f1_score(ground, prediction, average='weighted')
    cm = metrics.confusion_matrix(ground, prediction)

    return precision, recall, f1_score, cm
