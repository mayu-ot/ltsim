import numpy as np


def convert_cxcywy_to_ltrb(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding box format from [cx, cy, w, h] to [l, t, r, b]
    Args:
        bbox (np.ndarray): bounding box in [cx, cy, w, h]
    Returns:
        np.ndarray: bounding box in [l, t, r, b]
    """
    cx, cy, w, h = bbox.T
    l = cx - w / 2.0
    t = cy - h / 2.0
    r = cx + w / 2.0
    b = cy + h / 2.0
    return l, t, r, b


def compute_iou(
    box_1: np.ndarray,
    box_2: np.ndarray,
    generalized: bool = False,
) -> float:
    """Compute IoU between two boxes
    Args:
        box_1 (np.ndarray): box 1 [[cx, cy, w, h], ...]
        box_2 (np.ndarray): box 2 [[cx, cy, w, h], ...]
        generalized (bool, optional): If True, compute generalized IoU.
            Defaults to False.
    Returns:
        np.ndarray: NxM IoU array
    """
    # split into l, t, r, b
    l1, t1, r1, b1 = convert_cxcywy_to_ltrb(box_1)
    l2, t2, r2, b2 = convert_cxcywy_to_ltrb(box_2)

    n = len(box_1)
    m = len(box_2)

    iou_mat = np.zeros((n, m))

    # area
    area_1 = (r1 - l1) * (b1 - t1)
    area_2 = (r2 - l2) * (b2 - t2)

    for i in range(n):
        for j in range(m):
            # intersection
            l_max = np.maximum(l1[i], l2[j])
            r_min = np.minimum(r1[i], r2[j])
            t_max = np.maximum(t1[i], t2[j])
            b_min = np.minimum(b1[i], b2[j])
            # check if there is no intersection
            has_intersection = (l_max < r_min) & (t_max < b_min)
            area_intersection = (
                (r_min - l_max) * (b_min - t_max) if has_intersection else 0.0
            )

            area_union = area_1[i] + area_2[j] - area_intersection
            iou = area_intersection / area_union

            if not generalized:
                iou_mat[i, j] = iou
                continue

            # outer region
            l_min = np.minimum(l1[i], l2[j])
            r_max = np.maximum(r1[i], r2[j])
            t_min = np.minimum(t1[i], t2[j])
            b_max = np.maximum(b1[i], b2[j])
            area_conv_hull = (r_max - l_min) * (b_max - t_min)
            giou = iou - (area_conv_hull - area_union) / area_conv_hull
            iou_mat[i, j] = giou

    return iou_mat
