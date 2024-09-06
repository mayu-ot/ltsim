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
    if docsim_on:
        results["docsim"] = compute_doc_sim_for_layout_pair(layout1, layout2)
    if maximum_iou_on:
        results["maximum_iou"] = compute_maximum_iou_for_layout_pair(layout1, layout2)
    if mean_iou_on:
        results["mean_iou"] = compute_mean_iou_for_layout_pair(layout1, layout2)
    if docemd_on:
        results["docemd"] = compute_docemd_for_layout_pair(layout1, layout2)
    if ltsim_on:
        results["ltsim"] = compute_ltsim_between_layout(layout1, layout2)
    return results


def save_canvas(key):
    if key == "prev_layout2":
        if canvas_result2.json_data is not None:
            st.session_state[key] = canvas_result2.json_data
    elif key == "prev_layout3":
        if canvas_result3.json_data is not None:
            st.session_state[key] = canvas_result3.json_data


color_radio = st.sidebar.radio("Pick a color", ("🟥", "🟦", "🟩"))
if color_radio == "🟥":
    label_color = "rgba(255, 0, 0, 0.3)"
elif color_radio == "🟦":
    label_color = "rgba(0, 0, 255, 0.3)"
elif color_radio == "🟩":
    label_color = "rgba(0, 255, 0, 0.3)"
else:
    raise RuntimeError("Unknown color")


label = st.sidebar.text_input("Label", "Default")
mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"

docsim_on = st.sidebar.checkbox("DocSIM", False)
maximum_iou_on = st.sidebar.checkbox("MaxIoU", False)
mean_iou_on = st.sidebar.checkbox("MeanIoU", False)
docemd_on = st.sidebar.checkbox("DocEMD", False)
ltsim_on = st.sidebar.checkbox("LTsim", False)


if "prev_layout1" not in st.session_state:
    st.session_state["prev_layout1"] = {}
if "prev_layout2" not in st.session_state:
    st.session_state["prev_layout2"] = {}
if "prev_layout3" not in st.session_state:
    st.session_state["prev_layout3"] = {}

col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 3])

with col1:
    canvas_result1 = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        background_color="#fff",
        width=canvas_w,
        height=canvas_h,
        drawing_mode=mode,
        initial_drawing=None,
        key="query",
    )

with col2:
    synch_1 = st.checkbox(
        "🔗",
        True,
        on_change=save_canvas,
        args=["prev_layout2"],
        key="synch_1",
    )

with col3:
    canvas_result2 = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        background_color="#fff",
        width=canvas_w,
        height=canvas_h,
        drawing_mode=mode,
        initial_drawing=(
            canvas_result1.json_data if synch_1 else st.session_state["prev_layout2"]
        ),
        key="layout_1",
    )

with col4:
    synch_2 = st.checkbox(
        "🔗",
        True,
        on_change=save_canvas,
        args=["prev_layout3"],
        key="synch_2",
    )

with col5:
    canvas_result3 = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        background_color="#fff",
        width=canvas_w,
        height=canvas_h,
        drawing_mode=mode,
        initial_drawing=(
            canvas_result2.json_data if synch_2 else st.session_state["prev_layout3"]
        ),
        key="layout_2",
    )

layout1 = get_layout_from_canvas(canvas_result1)
layout2 = get_layout_from_canvas(canvas_result2)
layout3 = get_layout_from_canvas(canvas_result3)

if layout1 is not None and layout2 is not None and layout3 is not None:
    data = {"L1 vs L2": [], "L2 vs L3": [], "L1 vs L3": [], "measure": []}
    result_l1_l2 = compute_measures(layout1, layout2)
    result_l1_l3 = compute_measures(layout1, layout3)
    result_l2_l3 = compute_measures(layout2, layout3)
    for key in result_l1_l2.keys():
        data["L1 vs L2"].append(result_l1_l2[key])
        data["L2 vs L3"].append(result_l2_l3[key])
        data["L1 vs L3"].append(result_l1_l3[key])
        data["measure"].append(key)

    # visualize measure values with grouped bar chart
    data = pd.DataFrame(data)
    data = data.set_index("measure")
    st.plotly_chart(px.bar(data, barmode="group", range_y=[0, 3]))

# save canvas button
if st.button("Save"):
    colors = color_palette("colorblind")[:3]
    colors = [np.asarray(c) * 255 for c in colors]

    draw_cnf = {"colors": colors, "labels": [0, 1, 2]}
    f, axes = plt.subplots(1, 3, figsize=(12, 4))
    if layout1 is not None:
        ax = layout_to_svg_image(
            layout1,
            draw_cnf,
            size=(canvas_h, canvas_w),
            show_labels=False,
            ax=axes[0],
        )
        ax.set_title("Query")
    if layout2 is not None:
        ax = layout_to_svg_image(
            layout2,
            draw_cnf,
            size=(canvas_h, canvas_w),
            show_labels=False,
            ax=axes[1],
        )
        ax.set_title("Layout 1")
    if layout3 is not None:
        ax = layout_to_svg_image(
            layout3,
            draw_cnf,
            size=(canvas_h, canvas_w),
            show_labels=False,
            ax=axes[2],
        )
        ax.set_title("Layout 2")

    plt.savefig("figs/canvas.pdf", bbox_inches="tight")
