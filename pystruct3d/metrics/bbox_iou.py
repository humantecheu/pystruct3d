# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
# IoU core (iou_batch, match_iou_stats, and private helpers) ported from
# github.com/cv4aec/3d-matching-eval (https://github.com/cv4aec/3d-matching-eval).
# Changes vs original:
#   - Cost matrix uses proper L2 distance; original accidentally used only the
#     x-component due to a `[0]` indexing bug in calculate_cost_matrix.
#   - numba, logging, and config dependencies removed; thresholds are parameters.
#   - RigidRegistration (a verbatim pycpd copy) is not ported; add `pycpd` as a
#     dependency and use pycpd.RigidRegistration directly if CPD alignment is needed.

from contextlib import suppress

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError

from pystruct3d.bbox.bbox import BBox
from pystruct3d.bbox.utils import bbox_list2array
from pystruct3d.testing import create_bbox_lists
from pystruct3d.visualization import Visualizer

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _polygon_clip(subject: list, clip: list) -> list | None:
    """Sutherland-Hodgman polygon clipping.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subject: list of [x, y] points — any polygon.
        clip: list of [x, y] points — must be convex and counter-clockwise.

    Returns:
        Clipped polygon vertex list, or None if the intersection is empty.
    """

    def _inside(p: list) -> bool:
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) >= (cp2[1] - cp1[1]) * (
            p[0] - cp1[0]
        )

    def _intersect() -> list:
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output = subject
    cp1 = clip[-1]
    for cp2 in clip:
        inp, output = output, []
        s = inp[-1]
        for e in inp:
            if _inside(e):
                if not _inside(s):
                    output.append(_intersect())
                output.append(e)
            elif _inside(s):
                output.append(_intersect())
            s = e
        cp1 = cp2
        if not output:
            return None
    return output


def _poly_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Shoelace polygon area, broadcast over a leading batch dimension."""
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, 1, axis=-1), axis=-1)
        - np.sum(y * np.roll(x, 1, axis=-1), axis=-1)
    )


def _to_ccw(faces: np.ndarray) -> np.ndarray:
    """Reverse any clockwise face polygon in an (n, k, 2) array."""
    x, y = faces[..., 0], faces[..., 1]
    cw = (
        np.sum(x * np.roll(y, 1, axis=-1), axis=-1)
        - np.sum(y * np.roll(x, 1, axis=-1), axis=-1)
        > 0
    )
    faces[cw] = faces[cw][:, ::-1]
    return faces


def _pairwise_intersection_2d(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Pairwise 2D intersection areas via Sutherland-Hodgman + ConvexHull.

    Args:
        p1: (n, k, 2) face polygon array (counter-clockwise).
        p2: (m, k, 2) face polygon array (counter-clockwise).

    Returns:
        (n, m) array of intersection areas.
    """
    result = np.zeros((len(p1), len(p2)))
    for i in range(len(p1)):
        for j in range(len(p2)):
            try:
                inter = _polygon_clip(p1[i].tolist(), p2[j].tolist())
            except ZeroDivisionError:
                # Degenerate case: near-parallel edges produce a zero denominator.
                # Treat intersection area as 0 (conservative).
                continue
            if inter is not None:
                with suppress(QhullError, ValueError):
                    result[i, j] = ConvexHull(inter, qhull_options="QJ Pp").volume
    return result


# ---------------------------------------------------------------------------
# Array-level IoU (core)
# ---------------------------------------------------------------------------


def iou_batch(gt_corners: np.ndarray, pred_corners: np.ndarray) -> np.ndarray:
    """Batch 3D IoU between two sets of oriented bounding boxes.

    Corner ordering (same as ``pystruct3d.bbox.BBox.corner_points`` after
    ``order_points``):

    - corners[0:4]: bottom face (lower z), counter-clockwise in XY
    - corners[4:8]: top face  (upper z), counter-clockwise in XY

    CCW winding is enforced internally, so any consistent ordering works.

    Args:
        gt_corners: (n, 8, 3) ground-truth box corners.
        pred_corners: (m, 8, 3) predicted box corners.

    Returns:
        (n, m) float32 IoU matrix.
    """
    gt = gt_corners.astype(np.float64)
    pred = pred_corners.astype(np.float64)

    shift = min(float(gt.min()), float(pred.min()), 0.0)
    gt -= shift
    pred -= shift

    gface = _to_ccw(gt[:, :4, :2].copy())
    pface = _to_ccw(pred[:, :4, :2].copy())

    inter_area = _pairwise_intersection_2d(gface, pface)

    nz = np.argwhere(inter_area > 0)
    iou_3d = np.zeros((len(gt), len(pred)), dtype=np.float32)
    if len(nz) == 0:
        return iou_3d

    rows, cols = nz[:, 0], nz[:, 1]
    z_gt = gt[rows]
    z_pred = pred[cols]

    z_overlap = np.maximum(
        0.0,
        np.minimum(z_gt[:, 4, 2], z_pred[:, 4, 2])
        - np.maximum(z_gt[:, 0, 2], z_pred[:, 0, 2]),
    )

    inter_vol = inter_area[rows, cols] * z_overlap
    vol_gt = np.array([ConvexHull(pts, qhull_options="QJ Pp").volume for pts in z_gt])
    vol_pred = np.array([
        ConvexHull(pts, qhull_options="QJ Pp").volume for pts in z_pred
    ])

    union = vol_gt + vol_pred - inter_vol
    iou_3d[rows, cols] = np.where(union > 0, inter_vol / union, 0.0).astype(np.float32)
    np.clip(iou_3d, 0.0, 1.0, out=iou_3d)
    return iou_3d


def match_iou_stats(
    gt_corners: np.ndarray,
    pred_corners: np.ndarray,
) -> dict[str, float]:
    """IoU statistics for a set of structures via optimal IoU assignment.

    Builds the (n, m) IoU matrix with :func:`iou_batch`, solves the optimal
    assignment (LAP, maximising IoU), pads unmatched GT boxes with IoU = 0,
    and returns summary statistics.

    Args:
        gt_corners: (n, 8, 3) ground-truth box corners.
        pred_corners: (m, 8, 3) predicted box corners.

    Returns:
        Dict with keys ``"min"``, ``"max"``, ``"mean"``, ``"median"``, ``"std"``.
    """
    iou_matrix = iou_batch(gt_corners, pred_corners)
    rows, cols = linear_sum_assignment(iou_matrix, maximize=True)
    matched = iou_matrix[rows, cols].tolist()

    n_unmatched = len(gt_corners) - len(rows)
    if n_unmatched > 0:
        matched.extend([0.0] * n_unmatched)

    arr = np.asarray(matched, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


# ---------------------------------------------------------------------------
# BBox-object API (delegates to array core above)
# ---------------------------------------------------------------------------


def bbox_iou(bbox_1: BBox, bbox_2: BBox) -> float:
    """3D IoU between two bounding boxes.

    Args:
        bbox_1: first bounding box.
        bbox_2: second bounding box.

    Returns:
        IoU in [0, 1].
    """
    gt = bbox_1.corner_points[np.newaxis]
    pd = bbox_2.corner_points[np.newaxis]
    return float(iou_batch(gt, pd)[0, 0])


def mean_bbox_iou(
    groundtruth_bbox_list: list[BBox],
    predicted_bbox_list: list[BBox],
) -> float:
    """Mean IoU between two lists of bounding boxes via optimal assignment.

    Args:
        groundtruth_bbox_list: ground-truth bounding boxes.
        predicted_bbox_list: predicted bounding boxes.

    Returns:
        Mean IoU of the optimal assignment.
    """
    gt_c = bbox_list2array(groundtruth_bbox_list)
    pd_c = bbox_list2array(predicted_bbox_list)
    return match_iou_stats(gt_c, pd_c)["mean"]


def main() -> None:
    gt_boxes, pd_boxes = create_bbox_lists()
    Visualizer().add_bbox(gt_boxes, color=[1, 0, 0]).add_bbox(
        pd_boxes, color=[0, 0, 1]
    ).show()
    print(mean_bbox_iou(gt_boxes, pd_boxes))  # noqa: T201


if __name__ == "__main__":
    main()
