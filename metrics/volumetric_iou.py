import numpy as np
from typing import List, Tuple

def voxelize_pointcloud(
        pointcloud: np.ndarray,
        volume_limits: Tuple[np.ndarray, np.ndarray],
        voxel_size: float
    ) -> np.ndarray:
    min_vals, max_vals = volume_limits
    indices = ((pointcloud - min_vals) / voxel_size).astype(int)
    volume_dims = np.ceil((max_vals - min_vals) / voxel_size).astype(int)
    x_dim = volume_dims[0]
    y_dim = volume_dims[1]
    
    unravelled_indices = (
        indices[:, 2] * x_dim * y_dim +
        indices[:, 1] * x_dim +
        indices[:, 0]
    )
    
    return np.unique(unravelled_indices)

def volumetric_iou(
        groundtruth_pc: np.ndarray,
        predicted_pc: np.ndarray,
        voxel_size: float=0.01
    ) -> Tuple[float, float]:
    min_vals = np.floor(
        np.minimum(np.min(groundtruth_pc, axis=0), np.min(predicted_pc, axis=0))
        ).astype(int)
    max_vals = np.ceil(
        np.maximum(np.max(groundtruth_pc, axis=0), np.max(predicted_pc, axis=0))
        ).astype(int)
    volume_limits = (min_vals, max_vals)

    groundtruth_indices = voxelize_pointcloud(groundtruth_pc, volume_limits, voxel_size)
    predicted_indices = voxelize_pointcloud(predicted_pc, volume_limits, voxel_size)
    
    len_intersect = len(np.intersect1d(groundtruth_indices, predicted_indices))
    len_union = len(np.union1d(groundtruth_indices, predicted_indices))

    iou = len_intersect / len_union
    num_gt_voxels = len(groundtruth_indices)
    return iou, num_gt_voxels

def mean_volumetric_iou(classes_iou: List[Tuple[float, float]]) -> float:
    total_gt_voxels = 0
    for _, gt_voxels in classes_iou:
        total_gt_voxels += gt_voxels

    miou = 0
    for iou, gt_voxels in classes_iou:
        miou += (iou * gt_voxels / total_gt_voxels)
    
    return miou

def main() -> None:
    classes_iou = []
    for i in range(1, 4):
        # Generate two random point clouds with points in a 3D space ranging from [0, 20]
        pointcloud1 = np.random.uniform(low=0.0, high=20.0, size=(i * 1000000, 3))
        pointcloud2 = np.random.uniform(low=0.0, high=20.0, size=(i * 800000, 3))
        classes_iou.append(volumetric_iou(pointcloud1, pointcloud2, 0.1))
        print(f"Class_{i} IoU: {classes_iou[-1][0]}")
    
    print(f"mIoU: {mean_volumetric_iou(classes_iou)}")
    

if __name__ == "__main__":
    main()