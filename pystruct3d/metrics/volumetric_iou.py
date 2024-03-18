import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.metrics.generate_example import create_bbox_lists
from pystruct3d.metrics.voxelization_limits import voxelization_limits

# from pystruct3d.visualization.visualization import Visualization


def voxelize_bbox(
    bbox: BBox,
    volume_limits: tuple[np.ndarray, np.ndarray],
    voxel_size: float,
):
    """_summary_

    Args:
        bbox (BBox): _description_
        volume_limits (Tuple[np.ndarray, np.ndarray]): _description_
        voxel_size (float): _description_
    """
    min_vals, max_vals = volume_limits
    volume_dims = np.ceil((max_vals - min_vals) / voxel_size).astype(int)

    x_dim = volume_dims[0]
    y_dim = volume_dims[1]

    bbox_min, bbox_max = voxelization_limits(bbox)

    # Define voxel grid ranges
    x_range = np.arange(bbox_min[0], bbox_max[0], voxel_size) + voxel_size / 2
    y_range = np.arange(bbox_min[1], bbox_max[1], voxel_size) + voxel_size / 2
    z_range = np.arange(bbox_min[2], bbox_max[2], voxel_size) + voxel_size / 2

    # Create a 3D grid of midpoints
    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
    midpoints = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Check which midpoints are inside the convex hull
    points_in_bbox, _ = bbox.points_in_bbox_probability(midpoints)
    indices = ((points_in_bbox - min_vals) / voxel_size).astype(int)

    # fmt: off
    # TODO: check numpy builtin method for potential speedup
    # https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
    flattened_indices = (
        indices[:, 2] * x_dim * y_dim +
        indices[:, 1] * x_dim +
        indices[:, 0]
    )
    # fmt: on

    # vis = Visualization()
    # vis.bbox_geometry(bbox)
    # vis.point_cloud_geometry(points_in_bbox)
    # vis.visualize()

    return np.unique(flattened_indices)


def volumetric_iou(
    groundtruth_bboxes: list[BBox],
    predicted_bboxes: list[BBox],
    volume_limits: tuple[np.ndarray, np.ndarray] = None,
    voxel_size: float = 0.01,
):
    if volume_limits is None:
        volume_limits = voxelization_limits(groundtruth_bboxes, predicted_bboxes)

    # TODO: Vectorize and benchmark
    groundtruth_indices = np.array([])
    for bbox in groundtruth_bboxes:
        groundtruth_indices = np.union1d(
            groundtruth_indices, voxelize_bbox(bbox, volume_limits, voxel_size)
        )
    predicted_indices = np.array([])
    for bbox in predicted_bboxes:
        predicted_indices = np.union1d(
            predicted_indices, voxelize_bbox(bbox, volume_limits, voxel_size)
        )

    len_intersect = len(np.intersect1d(groundtruth_indices, predicted_indices))
    len_union = len(np.union1d(groundtruth_indices, predicted_indices))

    iou = len_intersect / len_union
    num_gt_voxels = len(groundtruth_indices)
    return iou, num_gt_voxels


def mean_volumetric_iou(classes_iou: list[tuple[float, float]]) -> float:
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


def main():
    bboxes_1, bboxes_2 = create_bbox_lists()
    bboxes_1[0].rotate(30)
    volumetric_iou(bboxes_1, bboxes_2)


if __name__ == "__main__":
    main()
