import numpy as np
from .utils import convert_cxcywy_to_ltrb
from typing import Optional
import multiprocessing


def compute_mean_iou_for_layout_pair(layout_1: dict, layout_2: dict):
    """
    Compute class-wise mean IoU
    Args:
        layout_1 (dict): layout 1
        layout_2 (dict): layout 2
    Returns:
        float: class-wise mean IoU
    """
    # get unique labels
    labels = np.unique(layout_1["category"] + layout_2["category"])
    # compute IoU for each label
    ious = []
    mask_size = 255

    # create mask of bboxes
    mask_1 = np.zeros((mask_size, mask_size))
    mask_2 = np.zeros((mask_size, mask_size))

    for label in labels:
        # get bboxes for each label
        bboxes_1 = [
            layout_1["bbox"][i]
            for i in range(len(layout_1["bbox"]))
            if layout_1["category"][i] == label
        ]
        bboxes_2 = [
            layout_2["bbox"][i]
            for i in range(len(layout_2["bbox"]))
            if layout_2["category"][i] == label
        ]

        # when label is only appear in one layout
        if len(bboxes_1) == 0 or len(bboxes_2) == 0:
            ious.append(0)
            continue

        for bbox in bboxes_1:
            l, t, r, b = convert_cxcywy_to_ltrb(np.asarray(bbox) * mask_size)
            mask_1[int(t) : int(b), int(l) : int(r)] = 1
        for bbox in bboxes_2:
            l, t, r, b = convert_cxcywy_to_ltrb(np.asarray(bbox) * mask_size)
            mask_2[int(t) : int(b), int(l) : int(r)] = 1

        # compute IoU
        intersection = np.logical_and(mask_1, mask_2)
        union = np.logical_or(mask_1, mask_2)
        if np.sum(union) == 0:  # can happen when elements are too small
            iou = 0
        else:
            iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)

        # clear mask
        mask_1[:] = 0
        mask_2[:] = 0

    return np.mean(ious)


def compute_meaniou(
    layouts_gt: list[dict],
    layouts_generated: list[dict],
    disable_parallel: bool = True,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute layout-to-layout similarity and average over layout pairs.
    Note that this is different from layouts-to-layouts similarity.
    """
    args = list(zip(layouts_gt, layouts_generated))
    if disable_parallel:
        scores = []
        for arg in args:
            scores.append(compute_mean_iou_for_layout_pair(*arg))
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(compute_mean_iou_for_layout_pair, args)
    return np.array(scores).mean()
