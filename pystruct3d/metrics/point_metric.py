import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from pystruct3d.bbox import utils
from pystruct3d.bbox.bbox import BBox


def vertex_precision_recall(
    gt_corners: np.ndarray,
    pred_corners: np.ndarray,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[float, dict[str, float]]:
    """Precision/recall/F1 at multiple vertex-distance thresholds.

    Flattens all 8 corners of each box into vertex lists, builds a pairwise
    L2 cost matrix, solves the optimal vertex assignment (LAP), and counts
    matched pairs within each threshold.

    Metric interpretation: a vertex is "matched" if the nearest assigned
    counterpart is within threshold metres.  Precision is the fraction of
    *predicted* vertices that are matched; recall is the fraction of *GT*
    vertices that are matched.

    Note: the original 3d-matching-eval cost matrix used only the x-component
    of the vertex difference (a ``[0]`` indexing bug).  This function uses
    proper L2 Euclidean distance, so scores will differ slightly from the
    original evaluator on non-trivial inputs.

    Args:
        gt_corners: (n, 8, 3) ground-truth box corners.
        pred_corners: (m, 8, 3) predicted box corners.
        thresholds: distance thresholds in the same units as the corners (metres).

    Returns:
        Dict keyed by threshold.  Each value is a dict with keys
        ``"precision"``, ``"recall"``, ``"f1"``, and ``"matched"`` (count).
    """
    gt_verts = gt_corners.reshape(-1, 3).astype(np.float32)
    pred_verts = pred_corners.reshape(-1, 3).astype(np.float32)

    result: dict[float, dict[str, float]] = {}

    if pred_verts.shape[0] == 0:
        for t in thresholds:
            result[t] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
        return result

    cost = cdist(gt_verts, pred_verts).astype(np.float32)
    rows, cols = linear_sum_assignment(cost)
    distances = cost[rows, cols]

    n_gt, n_pred = gt_verts.shape[0], pred_verts.shape[0]
    for t in thresholds:
        matched = int((distances < t).sum())
        prec = matched / n_pred
        rec = matched / n_gt
        denom = prec + rec
        f1 = (2.0 * prec * rec / denom) if denom > 0.0 else 0.0
        result[t] = {"precision": prec, "recall": rec, "f1": f1, "matched": matched}

    return result


def centroid_deviation(
    gt_bbox_list: list[BBox],
    pred_bbox_list: list[BBox],
    distance_upper_bound: float = 0.5,
) -> float:
    """Mean nearest-neighbour centroid deviation from GT to predicted boxes.

    For each GT centroid, finds the closest predicted centroid within
    ``distance_upper_bound`` metres (Euclidean). GT boxes with no match
    within that radius are excluded. Returns NaN if no GT box has a match.

    Args:
        gt_bbox_list: ground-truth bounding boxes.
        pred_bbox_list: predicted bounding boxes.
        distance_upper_bound: search radius in metres. Defaults to 0.5.

    Returns:
        Mean Euclidean centroid deviation over matched pairs, or NaN.
    """
    gt_centroids = np.mean(utils.bbox_list2array(gt_bbox_list), axis=1)
    pred_centroids = np.mean(utils.bbox_list2array(pred_bbox_list), axis=1)

    pred_kd = KDTree(pred_centroids)
    dists, _ = pred_kd.query(
        gt_centroids, k=1, p=2, distance_upper_bound=distance_upper_bound
    )

    matched = dists[np.isfinite(dists)]
    if len(matched) == 0:
        return float("nan")
    return float(np.mean(matched))
