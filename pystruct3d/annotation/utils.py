import e57
import numpy as np
import open3d as o3d

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
    print("Transfer labels ...")
    for ix, point in enumerate(labeled_pcd.points):
        _, idx, _ = unlabeled_tree.search_hybrid_vector_3d(point, 0.03, 100)
        # _, idx, _ = unlabeled_tree.search_knn_vector_3d(point, 1)
        # transfer label to new array
        if ix % 1000000 == 0:
            print(f"{ix:,}/{np.shape(labels_array)[0]:,} transferred ...")
        if idx:
            idx_arr = np.asarray(idx)
            new_labels[idx_arr] = labels_array[ix]

    return new_labels


def main():
    # labeled_pcd = o3d.io.read_point_cloud("/home/kaufmann/Desktop/labeled.pcd")

    # for LAS point clouds and numpy labels file
    # labels_arr = np.load("/home/kaufmann/Desktop/labels.npy")
    # labeled_pcd = las_utils.read_las_file("/home/kaufmann/Desktop/labeled.las")
    # unlabeled_pcd = las_utils.read_las_file("/home/kaufmann/Desktop/unlabeled.las")

    # when you use an ascii cloud that contains the labels you may use this code
    # just uncomment and comment the code above
    input_file = (
        "/home/kaufmann/Desktop/ADAC/20230804_ADAC_1_aligned_labeled_merged.asc"
    )
    input_arr = np.loadtxt(
        input_file,
        dtype=np.float32,
    )
    labeled_points = input_arr[:, 0:3]
    labeled_colors = input_arr[:, 3:6]
    labels_arr = input_arr[:, 6].flatten()
    print(f"Old labels: {np.unique(labels_arr)}")
    # combine arrays to open3d point cloud
    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(labeled_points)
    labeled_pcd.colors = o3d.utility.Vector3dVector(labeled_colors / 255)
    # o3d.visualization.draw_geometries([labeled_pcd])

    # open unlabeled point cloud
    # unlabeled_pcd = las_utils.read_las_file("/home/kaufmann/Desktop/unlabeled.las")
    unlabeled_file = "/home/kaufmann/Desktop/ADAC/20230727_ADAC_1_aligned.e57"
    unlabeled = e57.read_points(unlabeled_file)
    unlabeled_pcd = o3d.geometry.PointCloud()
    unlabeled_pcd.points = o3d.utility.Vector3dVector(unlabeled.points)
    unlabeled_pcd.colors = o3d.utility.Vector3dVector(unlabeled.color)
    print(f"Unlabeled {unlabeled_pcd}")
    print("Transferring labels ...")
    new_labels = transfer_labels(labeled_pcd, labels_arr, unlabeled_pcd)
    print(f"New labels: {np.unique(new_labels)}")

    ascii_pcd = np.hstack(
        (
            np.asarray(unlabeled_pcd.points, dtype=np.float32),
            np.asarray(unlabeled_pcd.colors, dtype=np.float32),
            new_labels.reshape(-1, 1),
        )
    )
    np.savetxt(f"{unlabeled_file[:-4]}_transfered_labels.txt", ascii_pcd, fmt="%.8e")


if __name__ == "__main__":
    main()
