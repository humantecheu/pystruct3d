# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
from __future__ import annotations

import logging
import time

import numpy as np
import open3d as o3d
from tqdm import tqdm

logger = logging.getLogger(__name__)


def transfer_labels(
    labeled_pcd: o3d.geometry.PointCloud,
    labels_array: np.ndarray,
    unlabeled_pcd: o3d.geometry.PointCloud,
    progress: bool = True,
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
        progress: show a tqdm progress bar. Defaults to True.

    Returns:
        Integer label array of shape (M,), one value per point in
        ``unlabeled_pcd``.  Unmatched points are assigned label 0.
    """
    n = len(labels_array)
    logger.info("Transferring labels: %d source points", n)
    t0 = time.perf_counter()

    new_labels = np.zeros((np.asarray(unlabeled_pcd.points).shape[0],), dtype=int)
    unlabeled_tree = o3d.geometry.KDTreeFlann(unlabeled_pcd)

    for point, label in tqdm(
        zip(labeled_pcd.points, labels_array, strict=False),
        total=n,
        desc="Transfer labels",
        disable=not progress,
    ):
        _, idx, _ = unlabeled_tree.search_hybrid_vector_3d(point, 0.03, 100)
        if idx:
            new_labels[np.asarray(idx)] = label

    logger.info("Label transfer done in %.1fs", time.perf_counter() - t0)
    return new_labels
