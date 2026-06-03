from __future__ import annotations

import numpy as np
import open3d as o3d
from tqdm import tqdm


def transfer_labels(
    labeled_pcd: o3d.geometry.PointCloud,
    labels_array: np.ndarray,
    unlabeled_pcd: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Transfer point-level labels from an annotated cloud to an unannotated one.

    For each point in ``labeled_pcd``, finds all neighbours within 3 cm in
    ``unlabeled_pcd`` (using a KD-tree hybrid search) and assigns them the
    source point's label.  Points in ``unlabeled_pcd`` that receive no
    neighbour assignment keep label 0 (background).

    This is an O(N) nearest-neighbour loop and can be slow for large clouds
    (tens of millions of points).  Consider downsampling both clouds first.

    Args:
        labeled_pcd: annotated source point cloud.
        labels_array: integer label array of shape (N,), one value per point
            in ``labeled_pcd``.
        unlabeled_pcd: target point cloud to receive the transferred labels.

    Returns:
        Integer label array of shape (M,), one value per point in
        ``unlabeled_pcd``.  Unmatched points are assigned label 0.
    """
    new_labels = np.zeros((np.asarray(unlabeled_pcd.points).shape[0],), dtype=int)
    unlabeled_tree = o3d.geometry.KDTreeFlann(unlabeled_pcd)

    for point, label in tqdm(
        zip(labeled_pcd.points, labels_array, strict=False),
        total=len(labels_array),
        desc="Transfer labels",
    ):
        _, idx, _ = unlabeled_tree.search_hybrid_vector_3d(point, 0.03, 100)
        if idx:
            new_labels[np.asarray(idx)] = label

    return new_labels
