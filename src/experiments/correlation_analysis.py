import json
from collections import defaultdict
from dataclasses import dataclass
from inspect import signature
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from pinjected import Design
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from experiments.utils import DataLoader, ReferenceLoader
from layout_eval.measures.docsim import compute_doc_sim_for_layout_pair
from layout_eval.measures.ltsim import (
    compute_emd_between_layout,
    provide_cost_mat_and_dual_var,
)
from layout_eval.measures.maximum_iou import (
    _group_layouts_by_labels,
    compute_maximum_iou_for_layout_pair,
)
from layout_eval.measures.mean_iou import compute_mean_iou_for_layout_pair
import click


def sample_similarity(data, measure_fncs: dict):
    result = defaultdict(list)
    layout_groups = _group_layouts_by_labels(data)
    for layout_group in tqdm(layout_groups.values()):
        for layout_pair in combinations(layout_group, 2):
            for k, measure_fnc in measure_fncs.items():
                # if measure_fnc args is annotated as dict
                if signature(measure_fnc).parameters["layout_1"].annotation == dict:
                    val = measure_fnc(layout_pair[0], layout_pair[1])
                else:
                    idx_1 = data.index(layout_pair[0])
                    idx_2 = data.index(layout_pair[1])
                    val = measure_fnc(idx_1, idx_2)
                result[k].append(val)
    return result


def compute_ltsim_for_layout_pair(
    layout_1: dict,
    layout_2: dict,
    mat_provider: callable = provide_cost_mat_and_dual_var,
):
    return np.exp(
        -compute_emd_between_layout(layout_1, layout_2, mat_provider=mat_provider)
    )


def fidsim_measure_builder(fid_feat_file: str):
    fid_feat = np.load(fid_feat_file)
    Y = pdist(fid_feat, "sqeuclidean")
    Y = squareform(Y)
    nu = 2 * np.median(Y) ** 2

    def compute_fidsim_for_layout_pair(layout_1: int, layout_2: int):
        return np.exp(-Y[layout_1, layout_2] / nu)

    return compute_fidsim_for_layout_pair


def get_measure_fncs(fid_feat_file: str):
    measure_fncs = {
        "docsim": compute_doc_sim_for_layout_pair,
        "maximum_iou": compute_maximum_iou_for_layout_pair,
        "ltsim": compute_ltsim_for_layout_pair,
        "mean_iou": compute_mean_iou_for_layout_pair,
        "fidsim": fidsim_measure_builder(fid_feat_file),
    }
    return measure_fncs


def create_report(result_file: str, log_dir: Path):
    data = json.load(open(result_file))
    df = pd.DataFrame(data)
    # reorder columns in the order of [docsim  maximum_iou  mean_iou fidsim ltsim]
    df = df[["docsim", "maximum_iou", "mean_iou", "fidsim", "ltsim"]]

    corr = df.corr(method="kendall")
    np.fill_diagonal(corr.values, np.nan)
    corr.loc["mean"] = corr.mean()
    corr = corr.style.highlight_max(axis=0, props="bfseries: ;")
    corr = corr.format("{:.3f}")
    latex_table = corr.to_latex(column_format="@{}rccccc@{}")

    # save latex table
    with open(log_dir / "corr_table.tex", "w") as f:
        f.write(latex_table)

    if len(df) > 1000:
        df = df.sample(1000)
    g = sns.pairplot(df)
    g.savefig(log_dir / "pairplot.pdf")


@dataclass
class CorrelationCheck:
    reference_loader: DataLoader
    measure_fncs: dict
    log_dir: str = "."
    experiment_runner: callable = sample_similarity

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

    def _save_results(self, results, out_file="results.json") -> None:
        json.dump(results, open(self.log_dir / out_file, "w"))

    def run(self) -> None:
        data = self.reference_loader.load()
        self.results = self.experiment_runner(data, self.measure_fncs)
        self._save_results(self.results, "sampled_similarity.json")
        create_report(self.log_dir / "sampled_similarity.json", self.log_dir)


@click.command()
@click.argument("dataset", type=str)
def run(dataset: str):
    """Correlation analysis between different similarity measures

    Args:
        dataset (str): dataset name [rico25, publaynet]
    """
    conf = (
        Design()
        .bind_instance(
            reference_file=f"data/datasets/{dataset}/val.json",
            log_dir=f"data/results/correlation_analysis/{dataset}/w_constraint",
            fid_feat_file=f"data/fid_feat/dataset/{dataset}_val.npy",
            experiment_runner=sample_similarity,
        )
        .bind_provider(measure_fncs=get_measure_fncs)
        .bind_class(
            reference_loader=ReferenceLoader,
        )
    )
    g = conf.to_graph()
    g.provide(CorrelationCheck).run()


if __name__ == "__main__":
    run()
