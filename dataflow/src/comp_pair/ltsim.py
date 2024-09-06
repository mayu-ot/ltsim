import numpy as np
import ot
from .utils import compute_iou


def provide_cost_mat_and_dual_var(layout_1: dict, layout_2: dict, g_iou=True):
    N = len(layout_1["category"])
    M = len(layout_2["category"])

    # compute IoU matrix
    iou_mat = compute_iou(
        np.array(layout_1["bbox"]),
        np.array(layout_2["bbox"]),
        generalized=g_iou,
    )
    if g_iou:
        iou_cost_mat = 1 - (iou_mat + 1) * 0.5
    else:
        iou_cost_mat = 1 - iou_mat

    # compute label difference matrix
    # 0 for same label, 1 for different label
    label_diff_mat = np.zeros((N, M))
    for i, c1 in enumerate(layout_1["category"]):
        for j, c2 in enumerate(layout_2["category"]):
            label_diff_mat[i, j] = 0 if c1 == c2 else 1

    cost_mat = iou_cost_mat * 0.5 + label_diff_mat * 0.5

    # define source and target distributions
    a = np.ones(N) / N
    b = np.ones(M) / M

    return a, b, cost_mat


def compute_lt_cost_between_layout(
    layout_1: dict,
    layout_2: dict,
    mat_provider: callable,
    post_process: callable = None,
    return_log: bool = False,
    use_sinkhorn: bool = False,
) -> float:
    """
    Compute OT cost between two layouts
    Args:
        layout_1 (dict): layout 1
        layout_2 (dict): layout 2
        cost_fnc (callable): pairwise-cost function
    """
    n_elem = len(layout_1["category"])
    m_elem = len(layout_2["category"])
    if n_elem == 0 or m_elem == 0:
        return 0.0

    # if layout dict is identical
    if layout_1 == layout_2:
        return 0.0

    a, b, cost_mat = mat_provider(layout_1, layout_2)
    if use_sinkhorn:
        raise RuntimeError("Sinkhorn is not supported yet")
        ot_mat = ot.sinkhorn(a, b, cost_mat, 1.0)
    else:
        ot_mat = ot.emd(a, b, cost_mat)  # return the OT matrix
    if post_process is not None:
        ot_mat = post_process(ot_mat)

    if return_log:
        return np.sum(ot_mat * cost_mat), {
            "ot_mat": ot_mat,
            "cost_mat": cost_mat,
        }

    return np.sum(ot_mat * cost_mat)


# convert cost func into kernel function
def convert_costfnc_to_kernel(cost_fnc: callable):
    kernel = lambda x, y: np.exp(-cost_fnc(x, y))
    return kernel
