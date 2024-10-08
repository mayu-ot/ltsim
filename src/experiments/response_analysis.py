import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
from pinjected import Design

from experiments.utils import DataLoader, ReferenceLoader
from layout_eval.measures.fid import calculate_frechet_distance
from layout_eval.measures.maximum_iou import compute_maximum_iou_for_layout_set
from layout_eval.measures.mmd import convert_emd_to_affinity, estimate_mmd


@click.group()
def cli():
    pass


def add_spatial_noise(
    layout: dict, spatial_noise_level: float = 0.2, noise_rate: float = 0.5
):
    for i in range(len(layout["category"])):
        # add noise to x and y in bbox [x, y, w, h]
        # generate an offset vector whose norm is at most spatial_noise_level
        # and add it to the current value
        if random.random() < noise_rate:
            length = random.uniform(0, spatial_noise_level)
            angle = random.uniform(0, 2 * np.pi)
            layout["bbox"][i][0] += length * np.cos(angle)
            layout["bbox"][i][1] += length * np.sin(angle)
    return layout


def add_category_noise(layout: dict, noise_rate: float = 0.5):
    for i in range(len(layout["category"])):
        if random.random() < noise_rate:
            label = layout["category"][i]
            # set to a random label that is not the current label
            new_label = random.choice([l for l in range(25) if l != label])
            layout["category"][i] = new_label
    return layout


def remove_or_add_elements(layout: dict, noise_rate: float = 0.5):
    if np.random.random() < noise_rate:
        if random.random() < 0.5:
            # remove element
            layout = remove_elements(layout, n_removal=1)
        else:
            # add element
            layout = add_elements(layout, n_addition=1)
        return layout


def add_elements(layout: dict, n_addition=1, noise_rate=0.5):
    if np.random.random() < noise_rate:
        for _ in range(n_addition):
            layout["category"].append(random.randint(0, 24))
            layout["bbox"].append(
                [
                    random.random(),
                    random.random(),
                    random.random(),
                    random.random(),
                ]
            )
    return layout


def remove_elements(layout, n_removal=1, noise_rate=0.5):
    if np.random.random() < noise_rate:
        for _ in range(n_removal):
            if len(layout["category"]) == 0:
                break
            i = random.randint(0, len(layout["category"]) - 1)
            layout["category"].pop(i)
            layout["bbox"].pop(i)
    return layout


def get_measure_fncs():
    return {
        "maximum_iou": lambda x, y: compute_maximum_iou_for_layout_set(x, y),
    }


def get_noise_fncs(element_level_noise_rate: float = 0.1):
    return [
        lambda x: add_spatial_noise(
            x, spatial_noise_level=0.1, noise_rate=element_level_noise_rate
        ),
        lambda x: add_category_noise(x, noise_rate=element_level_noise_rate),
    ]


@dataclass
class PerturbationInjector:
    """
    Generate perturbed datasets for response analysis
    """

    reference_loader: DataLoader
    n_runs: int = 10
    noise_fncs: list = None
    layout_level_noise_rate: float = 0.1
    out_dir: str = "."
    test_run: bool = False

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        data_dir = self.out_dir
        if not data_dir.exists():
            data_dir.mkdir()

    def add_noise_to_dataset(self, data: list):
        """
        Adds noise to the dataset by applying noise functions to layouts.

        Args:
            data (list): The original dataset of layouts.

        Returns:
            list: The dataset with noise added to some of the layouts.
        """
        data_w_noise = []
        for layout in data:
            if random.random() < self.layout_level_noise_rate:
                layout_w_noise = copy.deepcopy(layout)
                for noise_fnc in self.noise_fncs:
                    layout_w_noise = noise_fnc(layout_w_noise)
                data_w_noise.append(layout_w_noise)
            else:
                data_w_noise.append(copy.deepcopy(layout))
        return data_w_noise

    def run(self):
        data = self.reference_loader.load()
        if self.test_run:
            data = data[:500]
        for i in range(self.n_runs):
            data_w_noise = self.add_noise_to_dataset(data)
            json.dump(
                data_w_noise,
                open(self.out_dir / f"{i}.json", "w"),
            )


@dataclass
class Evaluator:
    """
    Evaluate the perturbed datasets generated by PerturbationInjector
    """

    reference_loader: DataLoader
    layouts_dir: str
    measure_fncs: dict = None
    log_dir: str = "."
    test_run: bool = False

    def __post_init__(self) -> None:
        self.layouts_dir = Path(self.layouts_dir)
        self.log_dir = Path(self.log_dir)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

    def run(self) -> list:
        """evaluate measure_fncs on the perturbed datasets

        Returns:
            list: N evaluation results. Each item is dict of measure_name: measure_value
        """
        data = self.reference_loader.load()
        self.results = []

        for file in self.layouts_dir.iterdir():
            data_w_noise = json.load(open(file))
            self.results.append(
                {k: fnc(data, data_w_noise) for k, fnc in self.measure_fncs.items()}
            )

        json.dump(self.results, open(self.log_dir / "eval_results.json", "w"))

        return self.results


@cli.command()
def gen_perturbed_dataset() -> None:
    """Generate perturbed datasets for response analysis"""
    conf = (
        Design()
        .bind_instance(
            reference_file="data/datasets/rico25/val.json",
            n_runs=10,
            layout_level_noise_rate=1.0,
            test_run=False,
        )
        .bind_class(
            reference_loader=ReferenceLoader,
        )
    )

    for elem_noise_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        conf = conf.bind_provider(
            noise_fncs=get_noise_fncs(elem_noise_rate),
        ).bind_instance(
            out_dir=f"data/results/response_analysis/small_spatial_and_label_noise/elem_noise_rate_{elem_noise_rate}/data",
        )
        g = conf.to_graph()
        g.provide(PerturbationInjector).run()


@cli.command()
def eval_perturbed_dataset() -> None:
    """Evaluate the perturbed datasets generated by gen_perturbed_dataset"""
    conf = (
        Design()
        .bind_instance(
            reference_file="data/datasets/rico25/val.json",
            measure_fncs=get_measure_fncs(),
            test_run=False,
        )
        .bind_class(
            reference_loader=ReferenceLoader,
        )
    )

    for elem_noise_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        conf = conf.bind_instance(
            layouts_dir=f"data/results/response_analysis/small_spatial_and_label_noise/elem_noise_rate_{elem_noise_rate}/data",
            log_dir=f"data/results/response_analysis/small_spatial_and_label_noise/elem_noise_rate_{elem_noise_rate}",
        )
        g = conf.to_graph()
        g.provide(Evaluator).run()


@cli.command()
@click.argument("data-dir", type=str)
@click.argument("dataset-feat", type=str)
@click.argument("out-dir", type=str)
def compute_fid(data_dir: str, dataset_feat: str, out_dir: str) -> None:
    """Compute FIDs

    Args:
        data_dir (str): a directory containing FID feat files
        dataset_feat (str): FID feat file of the reference dataset
        out_dir (str): output directory
    """
    ref_feat = np.load(dataset_feat)
    mu_1 = np.mean(ref_feat, axis=0)
    sig_1 = np.cov(ref_feat, rowvar=False)
    results = []
    for file in os.listdir(data_dir):
        print(file)
        if file.endswith(".npy"):
            feat = np.load(os.path.join(data_dir, file))
            mu_2 = np.mean(feat, axis=0)
            sig_2 = np.cov(feat, rowvar=False)

            fid = calculate_frechet_distance(mu_1, sig_1, mu_2, sig_2)
            results.append({"fid": fid})
    json.dump(results, open(os.path.join(out_dir, "fid.json"), "w"))


def _load_xx_xy(file: str) -> tuple[np.ndarray, np.ndarray]:
    """convert dataflow output csv file to xx and xy matrices

    Args:
        file (str): path to the csv file. Each row contains i, j, value.

    Returns:
        xx (np.ndarray): xx matrix. xx[i, j] is EMD between layout i and layout j in a perturbed dataset.
        xy (np.ndarray): xy matrix. xy[i, j] is EMD between layout i in a perturbed dataset and layout j in the reference dataset.
    """
    print("loading...", file)
    # each column contains i, j, value
    df = pd.read_csv(
        file,
        header=None,
        names=["i", "j", "value"],
    )
    df.i = df.i.astype(int)
    df.j = df.j.astype(int)
    N = (df.i.max() + 1) // 2

    xx = np.zeros((N, N))
    xy = np.zeros((N, N))
    for i, j, value in zip(df.i, df.j, df.value):
        if i < N and j < N:
            xx[i, j] = value
            xx[j, i] = value
        elif i >= N and j < N:
            xy[i - N, j] = value
    return xx, xy


@cli.command()
@click.argument("data-dir", type=str)
@click.argument("out-dir", type=str)
@click.argument("yy-file", type=str)
def compute_mmd(data_dir: str, out_dir: str, yy_file: str) -> None:
    """Compute MMD

    Args:
        data_dir (str): a directory containing dataflow output csv files
        out_dir (str): output directory
        yy_file (str): EMD matrix of the reference dataset
    """
    yy = np.load(yy_file)
    sigma = np.median(yy[np.triu_indices(len(yy))])
    yy = convert_emd_to_affinity(yy, sigma)

    result = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            xx, xy = _load_xx_xy(os.path.join(data_dir, file))
            xx = convert_emd_to_affinity(xx, sigma)
            xy = convert_emd_to_affinity(xy, sigma)
            mmd = estimate_mmd(xx, yy, xy)
            result.append({"mmd": mmd})
    json.dump(result, open(os.path.join(out_dir, "mmd.json"), "w"))


if __name__ == "__main__":
    cli()
