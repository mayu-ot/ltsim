import json
import os
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd

from experiments.utils import csvfile_to_array
from layout_eval.measures.fid import calculate_frechet_distance
from layout_eval.measures.maximum_iou import compute_maximum_iou_for_layout_set
from layout_eval.measures.mmd import convert_emd_to_affinity, estimate_mmd


def evaluate(file: str, reference_file: str) -> dict[str, float]:
    data = json.load(open(file))
    data = data["results"].values()
    ref_data = json.load(open(reference_file))
    ref_data = ref_data["annotations"].values()

    # compute maximum iou
    max_iou = compute_maximum_iou_for_layout_set(data, ref_data)
    return {"max_iou": max_iou}


def eval_fid(file: str, ref_feat_file: str) -> dict[str, float]:
    ref_feat = np.load(ref_feat_file)
    mu_1 = np.mean(ref_feat, axis=0)
    sig_1 = np.cov(ref_feat, rowvar=False)

    feat = np.load(file)
    mu_2 = np.mean(feat, axis=0)
    sig_2 = np.cov(feat, rowvar=False)

    fid = calculate_frechet_distance(mu_1, sig_1, mu_2, sig_2)
    return {"fid": fid}


def get_mmd_result(data_dir: str, yy_file: str) -> dict[str, list]:
    emd_yy = csvfile_to_array(yy_file)
    sigma = np.median(emd_yy[np.triu_indices(len(emd_yy), k=1)])
    yy = convert_emd_to_affinity(emd_yy, sigma)

    result = defaultdict(list)
    models = os.listdir(data_dir)
    for model in models:
        for i in range(3):
            base_path = os.path.join(data_dir, model, f"seed_{i}_run_0_")
            xx_file = base_path + "xx.csv"
            xy_file = base_path + "xy.csv"
            print(xx_file)
            print(xy_file)
            xx = csvfile_to_array(xx_file)
            xy = csvfile_to_array(xy_file)
            xx = convert_emd_to_affinity(xx, sigma)
            xy = convert_emd_to_affinity(xy, sigma)
            mmd = estimate_mmd(xx, yy, xy)
            result["model"].append(model)
            result["measure"].append("mmd")
            result["value"].append(mmd)
            result["file"].append(xx_file)
    return result


@click.command()
@click.argument("dataset", type=str)
@click.argument("save-file", type=str)
def run(dataset: str, save_file: str) -> None:
    if dataset == "rico":
        layout_dir = "data/results_unconditional/rico"
        fid_feat_dir = "data/fid_feat/results_unconditional/rico"
        reference_file = "data/datasets/rico25/test.json"
        ref_feat_file = "data/fid_feat/dataset/rico25_test.npy"
        emd_dir = "data/dataflow/outputs/results_unconditional/rico"
        yy_file = "data/dataflow/outputs/datasets/rico25/test_yy.csv"
    elif dataset == "publaynet":
        layout_dir = "data/results_unconditional/publaynet"
        fid_feat_dir = "data/fid_feat/results_unconditional/publaynet"
        reference_file = "data/datasets/publaynet/test.json"
        ref_feat_file = "data/fid_feat/dataset/publaynet_test.npy"
        emd_dir = "data/dataflow/outputs/results_unconditional/publaynet"
        yy_file = "data/dataflow/outputs/datasets/publaynet/test_yy.csv"

    result = {"model": [], "measure": [], "value": [], "file": []}

    # compute maximum iou
    for p in Path(layout_dir).glob("*/*.json"):
        print(p)
        model_name = p.parent.name
        eval_res = evaluate(p, reference_file)
        for k, v in eval_res.items():
            result["model"].append(model_name)
            result["measure"].append(k)
            result["value"].append(v)
            result["file"].append(str(p))

    # compute fid
    for p in Path(fid_feat_dir).glob("*/*.npy"):
        print(p)
        model_name = p.parent.name
        eval_res = eval_fid(p, ref_feat_file)

        result["model"].append(model_name)
        result["measure"].append("fid")
        result["value"].append(eval_res["fid"])
        result["file"].append(str(p))

    # compute mmd
    mmd_result = get_mmd_result(emd_dir, yy_file)

    # concat result and mmd_result
    for k, v in mmd_result.items():
        result[k].extend(v)

    save_path = Path(save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(save_file, index=False)


if __name__ == "__main__":
    run()
