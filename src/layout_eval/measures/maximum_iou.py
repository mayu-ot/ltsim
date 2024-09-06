import numpy as np
from collections import defaultdict
from .utils import compute_iou
from scipy.optimize import linear_sum_assignment
from itertools import chain
import logging

# Based on https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py


def compute_maximum_iou_for_layout_pair(
    layout_1: dict, layout_2: dict
) -> float:
    """Compute maximum IoU for two layouts
    Args:
        layout_1 (dict): normalized layout 1
        layout_2 (dict): normalized layout 2
        layout_1 and layout_2 have the following structure:
        {
            "category": list[int],
            "bbox": list[list[float]], # (center_x, center_y, width, height)
        }
        layout_1 and layout_2 are assumed to have the same types of elements.
    Returns:
        float: maximum IoU
    """
    if str(sorted(layout_1["category"])) != str(sorted(layout_2["category"])):
        return None

    if len(layout_1["category"]) == 0:
        return None

    bboxes_1 = np.array(layout_1["bbox"])
    bboxes_2 = np.array(layout_2["bbox"])
    category_1 = np.array(layout_1["category"])
    category_2 = np.array(layout_2["category"])
    N = len(bboxes_1)

    score = 0.0
    for l in set(category_1):
        _bboxes_1 = bboxes_1[np.where(category_1 == l)]
        _bboxes_2 = bboxes_2[np.where(category_2 == l)]
        iou = compute_iou(_bboxes_1, _bboxes_2)
        # note: maximize is supported only when scipy >= 1.4
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou_for_layout_group(
    layouts_1: list[dict], layouts_2: list[dict]
) -> np.ndarray:
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            compute_maximum_iou_for_layout_pair(layouts_1[i], layouts_2[j])
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def _group_layouts_by_labels(layouts: list[dict]) -> dict:
    group_by_labels = defaultdict(list)
    for layout in layouts:
        # set key as sorted labels
        key = str(sorted(layout["category"]))
        group_by_labels[key].append(layout)
    return group_by_labels


# todo: add logging to file
def log_stats(scores, layouts_1, layouts_2):
    logging.info(
        f"Maximum IoU: {np.mean(scores):.4f} "
        f"(#layout_1: {len(layouts_1)}, #layout_2: {len(layouts_2)}, #pairs evaluated: {len(scores)})"
    )


def compute_maximum_iou_for_layout_set(
    layouts_1: list[dict],
    layouts_2: list[dict],
    disable_parallel: bool = True,
    logging_func=None,
) -> float:
    """Compute maximum IoU for two lists of layouts
    Args:
        layouts_1 (list[dict]): list of normalized layouts 1
        layouts_2 (list[dict]): list of normalized layouts 2
        layouts_1 and layouts_2 have the following structure:
        [
            {
            "category": list[int],
            "bbox": list[list[float]], # (center_x, center_y, width, height)
            }
        ...
        ]
    Returns:
            float: maximum IoU [Kikuchi+, ACMMM'21]
    """
    # group layouts that has the same labels
    group_1_by_labels = _group_layouts_by_labels(layouts_1)
    group_2_by_labels = _group_layouts_by_labels(layouts_2)

    # get shared keys in both groups
    keys = set(group_1_by_labels.keys()) & set(group_2_by_labels.keys())
    args = [(group_1_by_labels[k], group_2_by_labels[k]) for k in keys]

    if disable_parallel:
        scores = [__compute_maximum_iou_for_layout_group(*arg) for arg in args]
    else:
        from joblib import Parallel, delayed

        scores = Parallel(n_jobs=-1)(
            [
                delayed(__compute_maximum_iou_for_layout_group)(*arg)
                for arg in args
            ]
        )
    scores = np.asarray(list(chain.from_iterable(scores)))

    if logging_func is not None:
        logging_func(scores, layouts_1, layouts_2)

    return scores.mean()
