import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.bbox.utils import bbox_list2array


def _set_iou(a_indices: np.ndarray, b_indices: np.ndarray) -> float:
    """IoU of two sorted flat voxel index arrays (set intersection / union)."""
    len_intersect = len(np.intersect1d(a_indices, b_indices))
    len_union = len(np.union1d(a_indices, b_indices))
    return len_intersect / len_union if len_union > 0 else 0.0


def _weighted_mean_iou(classes_iou: list[tuple[float, int]]) -> float:
    """GT-voxel-weighted mean IoU across classes.

    Args:
        classes_iou: list of (iou, num_gt_voxels) per class.

    Returns:
        Mean IoU weighted by the number of GT voxels per class.
    """
    total_gt_voxels = sum(gt_voxels for _, gt_voxels in classes_iou)
    return sum(iou * gt_voxels / total_gt_voxels for iou, gt_voxels in classes_iou)


def pointcloud_limits(pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the integer-snapped AABB of a point cloud.

    Args:
        pointcloud: array of shape (n, 3).

    Returns:
        min_values: floor of per-axis minimum, shape (3,).
        max_values: ceil of per-axis maximum, shape (3,).
    """
    assert pointcloud.shape[1] == 3, "pointcloud must be of shape nx3"
    min_values = np.floor(np.min(pointcloud, axis=0)).astype(int)
    max_values = np.ceil(np.max(pointcloud, axis=0)).astype(int)
    return min_values, max_values


def bbox_limits(bboxes_list: list[BBox]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the integer-snapped AABB enclosing a list of bounding boxes.

    Args:
        bboxes_list: list of BBox objects.

    Returns:
        min_vals: floor of the global per-axis minimum, shape (3,).
        max_vals: ceil of the global per-axis maximum, shape (3,).
    """
    bbox_array = bbox_list2array(bboxes_list)
    min_vals = np.floor(np.min(bbox_array.reshape(-1, 3), axis=0)).astype(int)
    max_vals = np.ceil(np.max(bbox_array.reshape(-1, 3), axis=0)).astype(int)
    return min_vals, max_vals


def voxelization_limits(
    *args: np.ndarray | BBox | list[BBox],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global voxel grid extents across any mix of point clouds and bounding boxes.

    Accepts any number of positional arguments, each of which can be a point cloud
    (np.ndarray of shape (n, 3)), a single BBox, or a list of BBox objects. Returns
    the tightest integer-snapped AABB that contains all inputs.

    Args:
        *args: any combination of np.ndarray, BBox, or list[BBox].

    Returns:
        limits_min: global per-axis minimum, shape (3,).
        limits_max: global per-axis maximum, shape (3,).

    Raises:
        TypeError: if an argument is not np.ndarray, BBox, or list[BBox].
    """
    limits_list = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            limits_list.append(pointcloud_limits(arg))
        elif isinstance(arg, BBox):
            limits_list.append(bbox_limits([arg]))
        elif isinstance(arg, list) and all(isinstance(item, BBox) for item in arg):
            limits_list.append(bbox_limits(arg))
        else:
            raise TypeError("Arguments can either be of type np.ndarray or List[BBox]")

    min_list, max_list = zip(*limits_list, strict=False)
    limits_min = np.min(np.array(min_list), axis=0).astype(int)
    limits_max = np.max(np.array(max_list), axis=0).astype(int)
    return limits_min, limits_max
