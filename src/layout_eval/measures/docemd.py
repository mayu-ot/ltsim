import numpy as np
from .utils import convert_cxcywy_to_ltrb
import ot


def compute_docemd_for_layout_pair(layout1, layout2, penalty=1.0):
    """
    Compute class-wise mean IoU
    Args:
        layout1 (dict): layout 1
        layout2 (dict): layout 2
        penalty (float): penalty for mismatching class
    Returns:
        float: class-wise mean IoU

        (original) https://github.com/Arking1995/simplified_diffusion/blob/d20dfde0825e4e287aa4dd78c8ecd4313f34e55b/evaluate_utlis.py#L195
    """
    # get unique labels
    labels = np.unique(layout1["category"] + layout2["category"])
    # compute IoU for each label
    emds = []
    mask_size = 64
    # create mask of bboxes
    mask_1 = np.zeros((mask_size, mask_size))
    mask_2 = np.zeros((mask_size, mask_size))

    for label in labels:
        # get bboxes for each label
        bboxes_1 = [
            layout1["bbox"][i]
            for i in range(len(layout1["bbox"]))
            if layout1["category"][i] == label
        ]
        bboxes_2 = [
            layout2["bbox"][i]
            for i in range(len(layout2["bbox"]))
            if layout2["category"][i] == label
        ]

        # note: label should appear either in layout1 or layout2 as labels are union of both layouts
        if (
            len(bboxes_1) == 0 or len(bboxes_2) == 0
        ):  # when label is only appear in one layout
            emds.append(penalty)
            continue

        for bbox in bboxes_1:
            l, t, r, b = convert_cxcywy_to_ltrb(np.asarray(bbox) * mask_size)
            mask_1[int(t) : int(b), int(l) : int(r)] = 1
        for bbox in bboxes_2:
            l, t, r, b = convert_cxcywy_to_ltrb(np.asarray(bbox) * mask_size)
            mask_2[int(t) : int(b), int(l) : int(r)] = 1

        img_1 = np.squeeze(mask_1)
        img_2 = np.squeeze(mask_2)
        xy_1 = np.argwhere(img_1)
        xy_2 = np.argwhere(img_2)
        xy_1 = xy_1 / mask_size  # 256 in original code
        xy_2 = xy_2 / mask_size

        m = xy_1.shape[0]
        n = xy_2.shape[0]
        weights_1 = np.ones((m,)) / m
        weights_2 = np.ones((n,)) / n
        cost = ot.max_sliced_wasserstein_distance(xy_1, xy_2, weights_1, weights_2)
        emds.append(cost)

        # clear mask
        mask_1[:] = 0
        mask_2[:] = 0

    return np.sum(emds)
