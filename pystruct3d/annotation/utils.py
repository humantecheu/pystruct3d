import numpy as np
import open3d as o3d
import pye57


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


def read_e57_as_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Read an E57 file and return the first scan as an Open3D point cloud."""
    e57_file = pye57.E57(path)
    scan = e57_file.read_scan(0, colors=True, transform=True)
    xyz = np.column_stack([scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if "colorRed" in scan:
        rgb = (
            np.column_stack([scan["colorRed"], scan["colorGreen"], scan["colorBlue"]])
            / 255.0
        )
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def main():
    # labeled_pcd = o3d.io.read_point_cloud("/path/to/labeled.pcd")
    # labels_arr = np.load("/path/to/labels.npy")

    # ASCII cloud with embedded labels (columns: X Y Z R G B label)
    input_file = (
        "/home/kaufmann/Desktop/ADAC/20230804_ADAC_1_aligned_labeled_merged.asc"
    )
    input_arr = np.loadtxt(input_file, dtype=np.float32)
    labeled_points = input_arr[:, 0:3]
    labeled_colors = input_arr[:, 3:6]
    labels_arr = input_arr[:, 6].flatten()
    print(f"Old labels: {np.unique(labels_arr)}")

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(labeled_points)
    labeled_pcd.colors = o3d.utility.Vector3dVector(labeled_colors / 255)

    unlabeled_file = "/home/kaufmann/Desktop/ADAC/20230727_ADAC_1_aligned.e57"
    unlabeled_pcd = read_e57_as_point_cloud(unlabeled_file)
    print(f"Unlabeled {unlabeled_pcd}")

    new_labels = transfer_labels(labeled_pcd, labels_arr, unlabeled_pcd)
    print(f"New labels: {np.unique(new_labels)}")

    ascii_pcd = np.hstack((
        np.asarray(unlabeled_pcd.points, dtype=np.float32),
        np.asarray(unlabeled_pcd.colors, dtype=np.float32),
        new_labels.reshape(-1, 1),
    ))
    np.savetxt(f"{unlabeled_file[:-4]}_transferred_labels.txt", ascii_pcd, fmt="%.8e")


if __name__ == "__main__":
    main()
