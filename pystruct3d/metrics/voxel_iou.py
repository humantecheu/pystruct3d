import numpy as np

from pystruct3d.metrics.voxelization_limits import (
    set_iou,
    weighted_mean_iou,
    voxelization_limits,
)


def voxelize_pointcloud(
    pointcloud: np.ndarray,
    volume_limits: tuple[np.ndarray, np.ndarray],
    voxel_size: float,
) -> np.ndarray:
    """Voxelize a point cloud and return occupied global voxel indices.

    Maps each point to a voxel bin in the global grid defined by
    ``volume_limits`` and returns the sorted unique set of occupied voxel indices.

    Args:
        pointcloud: array of shape (n, 3).
        volume_limits: (min_vals, max_vals) of the global voxel grid, shape (3,) each.
        voxel_size: voxel edge length in metres.

    Returns:
        Sorted array of unique flat voxel indices occupied by the point cloud.
    """
    min_vals, max_vals = volume_limits
    volume_dims = np.ceil((max_vals - min_vals) / voxel_size).astype(int)
    indices = ((pointcloud - min_vals) / voxel_size).astype(int)

    unravelled_indices = np.ravel_multi_index(
        (indices[:, 2], indices[:, 1], indices[:, 0]),
        (volume_dims[2], volume_dims[1], volume_dims[0]),
    )
    return np.unique(unravelled_indices)


def voxel_iou(
    groundtruth_pc: np.ndarray,
    predicted_pc: np.ndarray,
    volume_limits: tuple[np.ndarray, np.ndarray] | None = None,
    voxel_size: float = 0.01,
) -> tuple[float, int]:
    """Voxel-level IoU between two point clouds (3D semantic segmentation metric).

    Voxelizes each point cloud into a set of occupied voxel bins and computes
    IoU on those sets. Measures whether the same regions of space are occupied,
    regardless of point density — equivalent to 3D semantic segmentation IoU.

    Args:
        groundtruth_pc: ground-truth point cloud, shape (n, 3).
        predicted_pc: predicted point cloud, shape (m, 3).
        volume_limits: global voxel grid extents. Computed from both clouds if None.
        voxel_size: voxel edge length in metres. Defaults to 0.01.

    Returns:
        iou: intersection-over-union of the two voxel occupancy sets.
        num_gt_voxels: number of voxels occupied by the ground-truth cloud.
    """
    if volume_limits is None:
        volume_limits = voxelization_limits(groundtruth_pc, predicted_pc)

    groundtruth_indices = voxelize_pointcloud(groundtruth_pc, volume_limits, voxel_size)
    predicted_indices = voxelize_pointcloud(predicted_pc, volume_limits, voxel_size)

    iou = set_iou(groundtruth_indices, predicted_indices)
    num_gt_voxels = len(groundtruth_indices)
    return iou, num_gt_voxels


def mean_voxel_iou(classes_iou: list[tuple[float, int]]) -> float:
    """GT-voxel-weighted mean voxel IoU across classes.

    Args:
        classes_iou: list of (iou, num_gt_voxels) per class, as returned
            by :func:`voxel_iou`.

    Returns:
        Mean IoU weighted by the number of GT voxels per class.
    """
    return weighted_mean_iou(classes_iou)


def main() -> None:
    classes_iou = []
    for i in range(1, 4):
        pointcloud1 = np.random.uniform(low=0.0, high=20.0, size=(i * 1_000_000, 3))
        pointcloud2 = np.random.uniform(low=0.0, high=20.0, size=(i * 800_000, 3))
        classes_iou.append(
            voxel_iou(
                groundtruth_pc=pointcloud1, predicted_pc=pointcloud2, voxel_size=0.1
            )
        )
        print(f"Class_{i} IoU: {classes_iou[-1][0]}")

    print(f"mIoU: {mean_voxel_iou(classes_iou=classes_iou)}")


if __name__ == "__main__":
    main()
