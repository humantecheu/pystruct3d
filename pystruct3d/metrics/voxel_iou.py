from typing import List, Tuple

import numpy as np

from pystruct3d.metrics.voxelization_limits import voxelization_limits


def voxelize_pointcloud(
    pointcloud: np.ndarray,
    volume_limits: Tuple[np.ndarray, np.ndarray],
    voxel_size: float,
) -> np.ndarray:
    """_summary_

    Args:
        pointcloud (np.ndarray): _description_
        volume_limits (Tuple[np.ndarray, np.ndarray]): _description_
        voxel_size (float): _description_

    Returns:
        np.ndarray: _description_
    """
    min_vals, max_vals = volume_limits
    volume_dims = np.ceil((max_vals - min_vals) / voxel_size).astype(int)
    indices = ((pointcloud - min_vals) / voxel_size).astype(int)

    x_dim = volume_dims[0]
    y_dim = volume_dims[1]

    # fmt: off
    unravelled_indices = (
        indices[:, 2] * x_dim * y_dim +
        indices[:, 1] * x_dim +
        indices[:, 0]
    )
    # fmt: on

    return np.unique(unravelled_indices)


def voxel_iou(
    groundtruth_pc: np.ndarray,
    predicted_pc: np.ndarray,
    volume_limits: Tuple[np.ndarray, np.ndarray] = None,
    voxel_size: float = 0.01,
) -> Tuple[float, float]:
    """_summary_

    Args:
        groundtruth_pc (np.ndarray): _description_
        predicted_pc (np.ndarray): _description_
        volume_limits (Tuple[int, int]): _description_
        voxel_size (float, optional): _description_. Defaults to 0.01.

    Returns:
        Tuple[float, float]: _description_
    """
    if volume_limits is None:
        volume_limits = voxelization_limits(groundtruth_pc, predicted_pc)

    groundtruth_indices = voxelize_pointcloud(
        pointcloud=groundtruth_pc,
        volume_limits=volume_limits,
        voxel_size=voxel_size,
    )
    predicted_indices = voxelize_pointcloud(
        pointcloud=predicted_pc,
        volume_limits=volume_limits,
        voxel_size=voxel_size,
    )

    len_intersect = len(np.intersect1d(groundtruth_indices, predicted_indices))
    len_union = len(np.union1d(groundtruth_indices, predicted_indices))

    iou = len_intersect / len_union
    num_gt_voxels = len(groundtruth_indices)
    return iou, num_gt_voxels


def mean_voxel_iou(classes_iou: List[Tuple[float, float]]) -> float:
    """_summary_

    Args:
        classes_iou (List[Tuple[float, float]]): _description_

    Returns:
        float: _description_
    """
    total_gt_voxels = 0
    for _, gt_voxels in classes_iou:
        total_gt_voxels += gt_voxels

    miou = 0
    for iou, gt_voxels in classes_iou:
        miou += iou * gt_voxels / total_gt_voxels

    return miou


def main() -> None:
    classes_iou = []
    for i in range(1, 4):
        # Generate two random point clouds with points in a 3D space ranging from [0, 20]
        pointcloud1 = np.random.uniform(low=0.0, high=20.0, size=(i * 1000000, 3))
        pointcloud2 = np.random.uniform(low=0.0, high=20.0, size=(i * 800000, 3))
        classes_iou.append(
            voxel_iou(
                groundtruth_pc=pointcloud1,
                predicted_pc=pointcloud2,
                voxel_size=0.1,
            )
        )
        print(f"Class_{i} IoU: {classes_iou[-1][0]}")

    print(f"mIoU: {mean_voxel_iou(classes_iou=classes_iou)}")


if __name__ == "__main__":
    main()
