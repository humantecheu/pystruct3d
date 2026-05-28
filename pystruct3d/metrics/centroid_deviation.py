import numpy as np
from scipy.spatial import KDTree

from pystruct3d.bbox import utils


def get_centroids(bbox_list):
    bbox_array = utils.bbox_list2array(bbox_list)
    return np.mean(bbox_array, axis=1)


def deviation(
    bbox_list,
    gt_list,
    distance_upper_bound: float = 0.5,
) -> float:
    """Mean nearest-neighbour centroid deviation from GT to predicted boxes.

    For each GT centroid, finds the closest predicted centroid within
    ``distance_upper_bound`` metres (Euclidean). GT boxes with no match
    within that radius are excluded. Returns NaN if no GT box has a match.

    Args:
        bbox_list: predicted bounding boxes
        gt_list: ground-truth bounding boxes
        distance_upper_bound: search radius in metres. Defaults to 0.5.

    Returns:
        Mean Euclidean centroid deviation over matched pairs, or NaN.
    """
    bx_centroids = get_centroids(bbox_list)
    gt_centroids = get_centroids(gt_list)

    bx_kd_tree = KDTree(bx_centroids)
    dists, _ = bx_kd_tree.query(
        gt_centroids, k=1, p=2, distance_upper_bound=distance_upper_bound
    )

    matched = dists[np.isfinite(dists)]
    if len(matched) == 0:
        return float("nan")
    return float(np.mean(matched))


def main():
    rng = np.random.default_rng(42)
    bx_array = np.zeros((5, 8, 3))
    bx_list = utils.bbox_array2list(bx_array)
    gt_array = bx_array + rng.uniform(low=0.05, high=0.2)
    gt_list = utils.bbox_array2list(gt_array)
    print(deviation(bx_list, gt_list))


if __name__ == "__main__":
    main()
