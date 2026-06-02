import numpy as np
import open3d as o3d


def transfer_labels(
    labeled_pcd: o3d.geometry.PointCloud,
    labels_array: np.ndarray,
    unlabeled_pcd: o3d.geometry.PointCloud,
) -> np.ndarray:
    # label array for unlabeled pcd
    new_labels = np.zeros((np.asarray(unlabeled_pcd.points).shape[0],), dtype=int)

    # build kd tree
    unlabeled_tree = o3d.geometry.KDTreeFlann(unlabeled_pcd)

    # find nearest neighbour
    print("Transfer labels ...")
    for ix, point in enumerate(labeled_pcd.points):
        _, idx, _ = unlabeled_tree.search_hybrid_vector_3d(point, 0.03, 100)
        if ix % 1000000 == 0:
            print(f"{ix:,}/{np.shape(labels_array)[0]:,} transferred ...")
        if idx:
            idx_arr = np.asarray(idx)
            new_labels[idx_arr] = labels_array[ix]

    return new_labels
