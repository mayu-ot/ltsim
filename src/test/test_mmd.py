from layout_eval.measures.mmd import estimate_mmd
from layout_eval.measures.ltsim import (
    compute_emd_between_layout,
    provide_cost_mat_and_dual_var,
    convert_costfnc_to_kernel,
)
import json
import random
import numpy as np
import copy
from src.experiments.response_analysis import (
    add_elements,
    add_spatial_noise,
    add_category_noise,
    add_noise_to_dataset,
)

# def add_elements(layout, n_addition=1):
#     for _ in range(n_addition):
#         layout["category"].append(random.randint(0, 24))
#         layout["bbox"].append(
#             [
#                 random.random(),
#                 random.random(),
#                 random.random(),
#                 random.random(),
#             ]
#         )
#     return layout


# def add_noise_to_dataset(data, noise_rate, noise_funcs):
#     data_w_noise = []
#     for layout in data:
#         if random.random() < noise_rate:
#             layout_w_noise = copy.deepcopy(layout)
#             for noise_fnc in noise_funcs:
#                 layout_w_noise = noise_fnc(layout_w_noise)
#             data_w_noise.append(layout_w_noise)
#         else:
#             data_w_noise.append(copy.deepcopy(layout))
#     return data_w_noise


def load_test_data(N=100):
    data = json.load(open("data/cvpr2023_json/datasets/rico25/test.json"))
    data = list(data["annotations"].values())
    data = random.sample(data, N)
    return data


def test_selfsimilarity():
    data = load_test_data()

    oc_cost_fnc = lambda x, y: compute_emd_between_layout(
        x, y, provide_cost_mat_and_dual_var
    )
    kernel_fnc = convert_costfnc_to_kernel(oc_cost_fnc)

    mmd = estimate_mmd(data, data, kernel_fnc)
    assert np.abs(mmd) < 1e-2


def test_symmetry():
    n = 100
    data = load_test_data(2 * n)

    oc_cost_fnc = lambda x, y: compute_emd_between_layout(
        x, y, provide_cost_mat_and_dual_var
    )
    kernel_fnc = convert_costfnc_to_kernel(oc_cost_fnc)

    for i in range(n):
        layout1 = data[i]
        layout2 = data[i + n]
        diff = kernel_fnc(layout1, layout2) - kernel_fnc(layout2, layout1)
        assert np.abs(diff) < 1e-5


def test_response_to_noise():
    data = load_test_data()
    # compute self similarity
    oc_cost_fnc = lambda x, y: compute_emd_between_layout(
        x, y, provide_cost_mat_and_dual_var
    )
    kernel_fnc = convert_costfnc_to_kernel(oc_cost_fnc)
    selfsim_mmd = estimate_mmd(data, data, kernel_fnc)
    for noise_fnc in [add_elements, add_spatial_noise, add_category_noise]:
        data_w_noise = add_noise_to_dataset(data, 1.0, [noise_fnc])

        noise_data_mmd = estimate_mmd(data, data_w_noise, kernel_fnc)
        assert noise_data_mmd > selfsim_mmd
