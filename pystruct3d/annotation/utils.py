import laspy
import open3d as o3d
import numpy as np

from pystruct3d.utils import las_utils


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
    for ix, point in enumerate(labeled_pcd.points):
        _, idx, _ = unlabeled_tree.search_hybrid_vector_3d(point, 0.2, 1)
        idx_arr = np.asarray(idx)
        # transfer label to new array
        if idx_arr is not None:
            new_labels[idx_arr[0]] = labels_array[ix]

    return new_labels


def main():

    # labeled_pcd = o3d.io.read_point_cloud("/home/kaufmann/Desktop/labeled.pcd")
    labeled_arr = np.loadtxt("/home/kaufmann/Desktop/labeled.txt")
    labeled_points = labeled_arr[:, 0:3]
    labeled_colors = labeled_arr[:, 3:6]
    labels_arr = labeled_arr[:, 6].flatten()

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(labeled_points)
    labeled_pcd.colors = o3d.utility.Vector3dVector(labeled_colors / 255)
    o3d.visualization.draw_geometries([labeled_pcd])

    unlabeled_pcd = o3d.io.read_point_cloud("/home/kaufmann/Desktop/unlabeled.pcd")

    new_labels = transfer_labels(labeled_pcd, labels_arr, unlabeled_pcd)

    ascii_pcd = np.hstack((np.asarray(unlabeled_pcd.points), new_labels.reshape(-1, 1)))
    np.savetxt("/home/kaufmann/Desktop/transfered_labels.txt", ascii_pcd)


if __name__ == "__main__":
    main()
