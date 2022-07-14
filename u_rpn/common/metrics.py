""" Metrics to measure the performance of segmentation methods.

Written by: Miquel Miró Nicolau (UIB), 2022.
"""
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
from sklearn import metrics

from . import utils

Num = Union[int, float]


def relate_bbox_to_gt(bbox1, bbox2) -> Tuple[List[int], List[int], List[int]]:
    """Relates to sets of bounding boxes one to one with the IOU.

    Args:
        bbox1:
        bbox2:

    Returns:

    """
    overlaps = utils.compute_overlaps(bbox1, bbox2)
    relations = {}
    predictions = []

    bbox2_id = 0
    while bbox2_id < len(bbox2) and len(relations.keys()) < len(bbox1):
        maximums = np.argsort(overlaps[bbox2_id])[::-1]
        max_idx = 0

        while maximums[max_idx] in relations and max_idx < len(bbox1):
            max_idx += 1

        if max_idx < len(bbox1):
            relations[maximums[max_idx]] = bbox2_id
            predictions.append(int(overlaps[bbox2_id][maximums[max_idx]] > 0.7))
        else:
            predictions.append(0)

        bbox2_id += 1

    return list(relations.keys()), list(relations.values()), predictions


@lru_cache(maxsize=None)
def precision(true_positives: Num, false_positives: Num) -> float:
    """Calculate the precision through the confusion matrix info

    Args:
        true_positives:
        false_positives:

    Returns:

    """
    if true_positives != 0:
        return true_positives / (true_positives + false_positives)

    return 0.0


@lru_cache(maxsize=None)
def recall(true_positives: Num, false_negatives: Num) -> float:
    """Calculate the recall through the confusion matrix info

    Args:
        true_positives:
        false_negatives:

    Returns:

    """
    if true_positives != 0:
        return true_positives / (true_positives + false_negatives)
    return 0.0


@lru_cache(maxsize=None)
def f1_score(in_precision: Num, in_recall: Num) -> float:
    """Calculate the f1-score through the precision and the recall.

    Args:
        in_precision:
        in_recall:

    Returns:

    """
    if (in_precision + in_recall) != 0:
        return 2 * ((in_precision * in_recall) / (in_precision + in_recall))
    return 0.0


def basic_metrics(ground, prediction) -> Tuple[float, float, float]:
    """Calculate metrics for

    Args:
        ground:
        prediction:

    Returns:

    """
    cms = metrics.confusion_matrix(ground, prediction)

    fp = cms[1][0]
    fn = cms[0][1]
    tp = cms[1][1]

    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1_s = f1_score(prec, rec)

    return prec, rec, f1_s
