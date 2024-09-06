from layout_eval.measures.docemd import compute_docemd_for_layout_pair
from layout_eval.measures.ltsim import (
    compute_emd_between_layout,
    provide_cost_mat_and_dual_var,
)


def test_docemd_identical_layout():
    layout1 = {
        "bbox": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
        "category": [1, 2],
    }
    layout2 = {
        "bbox": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
        "category": [1, 2],
    }
    assert compute_docemd_for_layout_pair(layout1, layout2) == 0.0


def test_docemd_one_element_drop():
    layout1 = {
        "bbox": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
        "category": [1, 2],
    }
    layout2 = {
        "bbox": [[0.1, 0.1, 0.2, 0.2]],
        "category": [1],
    }
    assert compute_docemd_for_layout_pair(layout1, layout2) == 1.0


def test_docemd_one_element_addition():
    layout1 = {
        "bbox": [[0.42000000000000004, 0.275, 0.4, 0.29]],
        "category": [0],
    }
    layout2 = {
        "bbox": [
            [0.42000000000000004, 0.275, 0.4, 0.29],
            [0.36, 0.605, 0.22, 0.23],
            [0.715, 0.63, 0.19, 0.16],
        ],
        "category": [0, 1, 1],
    }
    assert compute_docemd_for_layout_pair(layout1, layout2) == 1.0


def test_oc_cost_identical_layout():
    layout1 = {
        "bbox": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
        "category": [1, 2],
    }
    layout2 = {
        "bbox": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
        "category": [1, 2],
    }
    mat_provider = lambda x, y: provide_cost_mat_and_dual_var(x, y, g_iou=True)
    cost = compute_emd_between_layout(layout1, layout2, mat_provider, post_process=None)
    assert cost == 0.0
