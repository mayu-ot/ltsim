import multiprocessing
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def __compute_bbox_sim(
    bboxes_1: np.ndarray,
    category_1: np.int64,
    bboxes_2: np.ndarray,
    category_2: np.int64,
    C_S: float = 2.0,
    C: float = 0.5,
) -> float:
    # bboxes from diffrent categories never match
    if category_1 != category_2:
        return 0.0

    cx1, cy1, w1, h1 = bboxes_1
    cx2, cy2, w2, h2 = bboxes_2

    delta_c = np.sqrt(np.power(cx1 - cx2, 2) + np.power(cy1 - cy2, 2))
    delta_s = np.abs(w1 - w2) + np.abs(h1 - h2)
    area = np.minimum(w1 * h1, w2 * h2)
    alpha = np.power(np.clip(area, 0.0, None), C)

    weight = alpha * np.power(2.0, -1.0 * delta_c - C_S * delta_s)
    return weight


def compute_doc_sim_for_layout_pair(
    layout_1: dict,
    layout_2: dict,
    max_diff_thresh: int = 3,
) -> float:
    bboxes_1 = np.array(layout_1["bbox"])
    bboxes_2 = np.array(layout_2["bbox"])
    category_1 = np.array(layout_1["category"])
    category_2 = np.array(layout_2["category"])

    N, M = len(bboxes_1), len(bboxes_2)
    if N >= M + max_diff_thresh or N <= M - max_diff_thresh:
        return 0.0

    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_bbox_sim(
                bboxes_1[i], category_1[i], bboxes_2[j], category_2[j]
            )
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)

    if len(scores[ii, jj]) == 0:
        # sometimes, predicted bboxes are somehow filtered.
        return 0.0
    else:
        return scores[ii, jj].mean()


def compute_docsim(
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
            scores.append(compute_doc_sim_for_layout_pair(*arg))
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(compute_doc_sim_for_layout_pair, args)
    return np.array(scores).mean()
