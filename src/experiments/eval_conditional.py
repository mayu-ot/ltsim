import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd
from pinjected import Design

from layout_eval.measures.docsim import compute_docsim
from layout_eval.measures.ltsim import compute_latsim_for_layout_set
from layout_eval.measures.maximum_iou import compute_maximum_iou_for_layout_set
from layout_eval.measures.mean_iou import compute_meaniou

import time


def evaluate(
    data: list, ref_data: list, measures: dict[str, callable], properties: dict = None
) -> dict:
    result = defaultdict(list)
    for measure_name, measure_fnc in measures.items():
        # start timer
        start = time.time()
        val = measure_fnc(data, ref_data)
        # end timer
        end = time.time()
        n_pair = len(data)
        print(
            f"{measure_name} took {end - start:.2f} seconds. avg time per pair: {(end - start) / n_pair * 1000:.2f} ms"
        )
        result["measure"].append(measure_name)
        result["value"].append(val)
        if properties is not None:
            for k, v in properties.items():
                result[k].append(v)
    return result


def get_measures() -> dict[str, callable]:
    measures = {
        "mean_iou": compute_meaniou,
        "doc_sim": compute_docsim,
        "max_iou": compute_maximum_iou_for_layout_set,
        "latsim": compute_latsim_for_layout_set,
    }
    return measures


@dataclass
class EvalConditional:
    data_dir: str
    ref_file: str
    measures: dict[str, callable]
    save_file: str
    test_run: bool = False

    def __post_init__(self):
        if self.save_file is not None:
            self.save_file = Path(self.save_file)
            self.save_file.parent.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("start evaluation")
        ref_data = json.load(open(self.ref_file))
        keys = list(ref_data["annotations"].keys())
        if self.test_run:
            keys = keys[:100]
        ref_layouts = [ref_data["annotations"][k] for k in keys]
        results = []
        print(self.data_dir)
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                print(file)
                task_name, model_name = root.split("/")[-2:]
                if task_name != "c":
                    continue
                if model_name not in [
                    "bart",
                    "blt",
                    "layoutdm",
                    "maskgit",
                    "vqdiffusion",
                ]:
                    continue
                file = os.path.join(root, file)
                pred_data = json.load(open(file))
                pred_layouts = [pred_data["results"][k] for k in keys]
                result = evaluate(
                    pred_layouts,
                    ref_layouts,
                    self.measures,
                    {"task": task_name, "model": model_name},
                )
                results.append(result)

        # concat results
        result = defaultdict(list)
        for r in results:
            for k, v in r.items():
                result[k].extend(v)

        if self.save_file is not None:
            pd.DataFrame(result).to_csv(self.save_file, index=False)


@click.command()
@click.argument("dataset", type=str)
@click.option("--test-run", is_flag=True)
@click.option("--dry-run", is_flag=True)
def run(dataset: str, test_run: bool = False, dry_run: bool = False):
    """conditional evaluation
    The script loads the results from data_dir and compares them with the reference data in ref_file.
    The evaluation results are saved in save_file.
    """
    if dataset == "rico":
        data_dir = "data/results_conditional/rico25"
        reference_file = "data/datasets/rico25/test.json"
        save_file = "data/results/eval_conditional/rico/result.csv"
    elif dataset == "publaynet":
        data_dir = "data/results_conditional/publaynet"
        reference_file = "data/datasets/publaynet/test.json"
        save_file = "data/results/eval_conditional/publaynet/result.csv"

    if dry_run:
        save_file = None

    conf = (
        Design()
        .bind_instance(
            data_dir=data_dir,
            ref_file=reference_file,
            measures=get_measures(),
            save_file=save_file,
            test_run=test_run,
        )
        .bind_provider(measures=get_measures)
    )
    g = conf.to_graph()
    g.provide(EvalConditional).run()


if __name__ == "__main__":
    run()
