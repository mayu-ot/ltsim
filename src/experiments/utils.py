import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# base class of data loaders
class DataLoader:
    def load(self):
        raise NotImplementedError


@dataclass
class PredictionLoader(DataLoader):
    prediction_file: str

    def load(self):
        return list(json.load(open(self.prediction_file))["results"].values())


@dataclass
class ReferenceLoader(DataLoader):
    reference_file: str

    def load(self):
        return list(json.load(open(self.reference_file))["annotations"].values())


def layout_to_svg_image(
    layout: dict,
    dataset_meta: dict,
    size: tuple = (256, 144),
    border_cfg: dict = None,
    show_labels: bool = True,
    ax=None,
):
    # create a matplotlib figure and draw layout on it
    if ax is None:
        ax = plt.figure(figsize=(size[0], size[1])).add_subplot(111)
    color_palette = dataset_meta["colors"]
    labels = dataset_meta["labels"]
    bbox = np.array(layout["bbox"])
    height, width = size

    for category, bbox in zip(layout["category"], layout["bbox"]):
        cx, cy, w, h = bbox
        cy = 1 - cy  # as matplotlib use bottom-left as origin
        x1 = (cx - w / 2.0) * width
        y1 = (cy - h / 2.0) * height
        w = w * width
        h = h * height
        color = np.asarray(color_palette[category]) / 255.0
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=True, color=color, alpha=0.5))
        if show_labels:
            ax.text(
                cx * width,
                cy * height,
                labels[category],
                color=(0, 0, 0),
                ha="center",
                va="center",
            )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")
    # add border
    if border_cfg is not None:
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                width,
                height,
                fill=False,
                color=border_cfg["color"],
                linewidth=border_cfg["width"],
            )
        )
    return ax


def csvfile_to_array(file, save_npy=True):
    if os.path.exists(file.replace(".csv", ".npy")):
        return np.load(file.replace(".csv", ".npy"))

    df = pd.read_csv(
        file,
        header=None,
        names=["i", "j", "value"],
    )
    N = int(df.i.max() + 1)
    M = int(df.j.max() + 1)

    emd = np.zeros((N, M))
    for i, j, value in df.values:
        emd[int(i), int(j)] = value

    if save_npy:
        np.save(file.replace(".csv", ".npy"), emd)
    return emd
