import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.testing import create_bbox_lists
from pystruct3d.metrics.voxelization_limits import (
    set_iou,
    weighted_mean_iou,
    voxelization_limits,
)


def voxelize_bbox(
    bbox: BBox,
    volume_limits: tuple[np.ndarray, np.ndarray],
    voxel_size: float,
) -> np.ndarray:
    """Voxelize a bounding box interior and return occupied global voxel indices.

    Generates a dense meshgrid of voxel midpoints within the bbox AABB, filters
    them with OBB containment, and maps each surviving midpoint to a flat index
    in the global voxel grid defined by ``volume_limits``.

    Args:
        bbox: bounding box to voxelize.
        volume_limits: (min_vals, max_vals) of the global voxel grid, shape (3,) each.
        voxel_size: voxel edge length in metres.

    Returns:
        Sorted array of unique flat voxel indices occupied by the bbox interior.
    """
    min_vals, max_vals = volume_limits
    volume_dims = np.ceil((max_vals - min_vals) / voxel_size).astype(int)

    bbox_min, bbox_max = voxelization_limits(bbox)

    x_range = np.arange(bbox_min[0], bbox_max[0], voxel_size) + voxel_size / 2
    y_range = np.arange(bbox_min[1], bbox_max[1], voxel_size) + voxel_size / 2
    z_range = np.arange(bbox_min[2], bbox_max[2], voxel_size) + voxel_size / 2

    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
    midpoints = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    points_inside, _ = bbox.points_in_bbox(midpoints)
    indices = ((points_inside - min_vals) / voxel_size).astype(int)

    flattened = np.ravel_multi_index(
        (indices[:, 2], indices[:, 1], indices[:, 0]),
        (volume_dims[2], volume_dims[1], volume_dims[0]),
    )
    return np.unique(flattened)


def volumetric_iou(
    groundtruth_bboxes: list[BBox],
    predicted_bboxes: list[BBox],
    volume_limits: tuple[np.ndarray, np.ndarray] | None = None,
    voxel_size: float = 0.01,
) -> tuple[float, int]:
    """Scene-level volumetric IoU between two sets of bounding boxes.

    Voxelizes every GT box into a single occupancy set and every predicted box
    into another, then computes IoU on those sets. This is a detection coverage
    metric at the scene level, not a per-box metric.

    Args:
        groundtruth_bboxes: ground-truth bounding boxes.
        predicted_bboxes: predicted bounding boxes.
        volume_limits: global voxel grid extents. Computed from both lists if None.
        voxel_size: voxel edge length in metres. Defaults to 0.01.

    Returns:
        iou: intersection-over-union of the two voxel occupancy sets.
        num_gt_voxels: number of voxels occupied by the ground-truth set.
    """
    if volume_limits is None:
        volume_limits = voxelization_limits(groundtruth_bboxes, predicted_bboxes)

    groundtruth_indices = np.unique(
        np.concatenate([
            voxelize_bbox(b, volume_limits, voxel_size) for b in groundtruth_bboxes
        ])
    )
    predicted_indices = np.unique(
        np.concatenate([
            voxelize_bbox(b, volume_limits, voxel_size) for b in predicted_bboxes
        ])
    )

    iou = set_iou(groundtruth_indices, predicted_indices)
    num_gt_voxels = len(groundtruth_indices)
    return iou, num_gt_voxels


def mean_volumetric_iou(classes_iou: list[tuple[float, int]]) -> float:
    """GT-voxel-weighted mean volumetric IoU across classes.

    Args:
        classes_iou: list of (iou, num_gt_voxels) per class, as returned
            by :func:`volumetric_iou`.

    Returns:
        Mean IoU weighted by the number of GT voxels per class.
    """
    return weighted_mean_iou(classes_iou)


def main() -> None:
    bboxes_1, bboxes_2 = create_bbox_lists()
    bboxes_1[0].rotate(30)
    volumetric_iou(bboxes_1, bboxes_2)


if __name__ == "__main__":
    main()
