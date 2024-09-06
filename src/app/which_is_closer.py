import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
import plotly.express as px

from layout_eval.measures.maximum_iou import (
    compute_maximum_iou_for_layout_pair,
)
from layout_eval.measures.ltsim import compute_ltsim_between_layout
from layout_eval.measures.docsim import compute_doc_sim_for_layout_pair
from layout_eval.measures.mean_iou import compute_mean_iou_for_layout_pair
from layout_eval.measures.docemd import compute_docemd_for_layout_pair
from experiments.utils import layout_to_svg_image
from matplotlib import pyplot as plt
from seaborn import color_palette

canvas_w = 100
canvas_h = 120


def display_labels(labels: list[str], ax: plt.Axes):
    for i, label in enumerate(labels):
        ax.text(
            0.5,
            0.9 - i * 0.15,
            f"{label}",
            ha="center",
            va="top",
            color="white",
            bbox=dict(
                boxstyle="round",
                fc=(0.56, 0.2, 0.65),
                ec=(0.56, 0.2, 0.65),
            ),
        )


def create_figure(layout1: dict, layout2: dict, layout3: dict, which_is_closer: dict):
    colors = color_palette("colorblind")[:3]
    colors = [np.asarray(c) * 255 for c in colors]

    draw_cnf = {"colors": colors, "labels": [0, 1, 2]}
    f, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    if layout1 is not None:
        ax = layout_to_svg_image(
            layout1,
            draw_cnf,
            size=(canvas_h, canvas_w),
            border_cfg={"color": "gray", "width": 1},
            show_labels=False,
            ax=axes[0],
        )
        ax.set_title("Layout A")
    if layout2 is not None:
        ax = layout_to_svg_image(
            layout2,
            draw_cnf,
            size=(canvas_h, canvas_w),
            border_cfg={"color": "gray", "width": 1},
            show_labels=False,
            ax=axes[1],
        )
        ax.set_title("Anchor")
    if layout3 is not None:
        ax = layout_to_svg_image(
            layout3,
            draw_cnf,
            size=(canvas_h, canvas_w),
            border_cfg={"color": "gray", "width": 1},
            show_labels=False,
            ax=axes[2],
        )
        ax.set_title("Layout B")

    measures_a = [
        measure for measure, a_or_b in which_is_closer.items() if a_or_b == "A"
    ]
    measures_ab = [
        measure for measure, a_or_b in which_is_closer.items() if a_or_b == "A=B"
    ]
    measures_b = [
        measure for measure, a_or_b in which_is_closer.items() if a_or_b == "B"
    ]

    display_labels(measures_a, axes[3])
    axes[3].set_title("A is closer")
    display_labels(measures_ab, axes[4])
    axes[4].set_title("A=B")
    display_labels(measures_b, axes[5])
    axes[5].set_title("B is closer")

    for ax in axes:
        ax.axis("off")
    return f


def get_cxcywh(object, size=(canvas_w, canvas_h)):
    l = object["left"]
    t = object["top"]
    w = object["width"] * object["scaleX"]
    h = object["height"] * object["scaleY"]
    cx = l + w / 2
    cy = t + h / 2
    cx = cx / size[0]
    cy = cy / size[1]
    w = w / size[0]
    h = h / size[1]
    return [cx, cy, w, h]


def convert_fill_to_label(fill):
    if fill == "rgba(255, 0, 0, 0.3)":
        return 0
    elif fill == "rgba(0, 0, 255, 0.3)":
        return 1
    elif fill == "rgba(0, 255, 0, 0.3)":
        return 2
    else:
        raise RuntimeError("Unknown fill")


def get_layout_from_canvas(canvas_result, coord_type="xywh"):
    if canvas_result.json_data is None:
        return None
    objects = canvas_result.json_data["objects"]
    layout = {
        "bbox": [],
        "category": [],
    }
    for object in objects:
        if object["type"] == "rect":
            bbox = get_cxcywh(object, size=(canvas_w, canvas_h))
            layout["bbox"].append(bbox)
            label = convert_fill_to_label(object["fill"])
            layout["category"].append(label)
    return layout


def compute_measures(layout1, layout2):
    results = {}
    results["DocSim"] = compute_doc_sim_for_layout_pair(layout1, layout2)
    results["Max.IoU"] = compute_maximum_iou_for_layout_pair(layout1, layout2)
    results["MeanIoU"] = compute_mean_iou_for_layout_pair(layout1, layout2)
    results["DocEMD"] = compute_docemd_for_layout_pair(layout1, layout2)
    results["LTSim"] = compute_ltsim_between_layout(layout1, layout2)
    return results


color_radio = st.sidebar.radio("Pick a color", ("🟥", "🟦", "🟩"))
if color_radio == "🟥":
    label_color = "rgba(255, 0, 0, 0.3)"
elif color_radio == "🟦":
    label_color = "rgba(0, 0, 255, 0.3)"
elif color_radio == "🟩":
    label_color = "rgba(0, 255, 0, 0.3)"
else:
    raise RuntimeError("Unknown color")

mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"

# # copy button
# src_layout = st.selectbox("src layout", ["A", "Anchor", "B"], key="src_layout")
# target_layout = st.selectbox(
#     "target layout", ["A", "Anchor", "B"], key="target_layout"
# )
# copy_button = st.button("Copy")
# if copy_button:
#     if src_layout == "A":
#         src_layout_data = canvas_a.json_data
#     elif src_layout == "Anchor":
#         src_layout_data = canvas_anchor.json_data
#     elif src_layout == "B":
#         src_layout_data = canvas_b.json_data
#     else:
#         pass

col1, col2, col3 = st.columns([3, 3, 3])

with col1:
    canvas_a = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        background_color="#fff",
        width=canvas_w,
        height=canvas_h,
        drawing_mode=mode,
        initial_drawing=None,
        key="layout_0",
    )
with col2:
    canvas_anchor = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        background_color="#fff",
        width=canvas_w,
        height=canvas_h,
        drawing_mode=mode,
        key="layout_1",
    )
with col3:
    canvas_b = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        background_color="#fff",
        width=canvas_w,
        height=canvas_h,
        drawing_mode=mode,
        key="layout_2",
    )

layout_a = get_layout_from_canvas(canvas_a)
layout_anchor = get_layout_from_canvas(canvas_anchor)
layout_b = get_layout_from_canvas(canvas_b)

if layout_a is not None and layout_anchor is not None and layout_b is not None:
    data = {"Anchor vs A": [], "Anchor vs B": [], "measure": []}
    res_anchor_a = compute_measures(layout_anchor, layout_a)
    res_anchor_b = compute_measures(layout_anchor, layout_b)
    which_is_closer = {}

    for key in res_anchor_a.keys():
        if res_anchor_a[key] is None or res_anchor_b[key] is None:
            continue
        if res_anchor_a[key] > res_anchor_b[key]:
            which_is_closer[key] = "A"
        elif res_anchor_a[key] < res_anchor_b[key]:
            which_is_closer[key] = "B"
        else:
            which_is_closer[key] = "A=B"

        if key == "DocEMD":
            # flip which_is_closer
            if which_is_closer[key] == "A":
                which_is_closer[key] = "B"
            elif which_is_closer[key] == "B":
                which_is_closer[key] = "A"
            else:
                pass

        data["Anchor vs A"].append(res_anchor_a[key])
        data["Anchor vs B"].append(res_anchor_b[key])
        data["measure"].append(key)

    # visualize measure values with grouped bar chart
    data = pd.DataFrame(data)
    data = data.set_index("measure")
    st.plotly_chart(px.bar(data, barmode="group", range_y=[0, 3]))

which_is_closer

f = create_figure(layout_a, layout_anchor, layout_b, which_is_closer)
st.pyplot(f)

# save canvas button
if st.button("Save"):
    save_id = f"{pd.Timestamp.now()}"
    f.savefig(f"figs/which_is_closer/{save_id}.pdf", bbox_inches="tight")
