import numpy as np

def voxelize_pointcloud(pointcloud, volume_limits, voxel_size):
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

def volumetric_iou(groundtruth_pc, predicted_pc, voxel_size=0.01):
    min_vals = np.floor(np.minimum(np.min(groundtruth_pc, axis=0), np.min(predicted_pc, axis=0))).astype(int)
    max_vals = np.ceil(np.maximum(np.max(groundtruth_pc, axis=0), np.max(predicted_pc, axis=0))).astype(int)

    groundtruth_indices = voxelize_pointcloud(groundtruth_pc, (min_vals, max_vals), voxel_size)
    predicted_indices = voxelize_pointcloud(predicted_pc, (min_vals, max_vals), voxel_size)
    
    len_intersect = len(np.intersect1d(groundtruth_indices, predicted_indices))
    len_union = len(np.union1d(groundtruth_indices, predicted_indices))

    return len_intersect / len_union


def main():
    # Generate two random point clouds with points in a 3D space ranging from [0, 50]
    pointcloud1 = np.random.uniform(low=0.0, high=50.0, size=(30000000, 3))
    pointcloud2 = np.random.uniform(low=0.0, high=50.0, size=(20000000, 3))
    print(volumetric_iou(pointcloud1, pointcloud2, 0.01))
    

if __name__ == "__main__":
    main()