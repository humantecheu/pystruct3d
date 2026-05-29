import numpy as np


def downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-downsample a point cloud and return the mapping back to the original.

    Each voxel cell is collapsed to its centroid. Returns both the downsampled
    points and an index array that maps every original point to its voxel's
    downsampled point, which can be used to propagate per-downsampled labels
    back to full resolution.

    Args:
        points: (n, 3) numpy array.
        voxel_size: Edge length of each cubic voxel.

    Returns:
        down_pts: (m, 3) centroid of each occupied voxel (m <= n).
        voxel_idx: (n,) integer array — voxel_idx[i] is the index in down_pts
            of the voxel that contains original point i.
    """
    min_coords = points.min(axis=0)
    grid_size = np.maximum(
        np.ceil((points.max(axis=0) - min_coords) / voxel_size).astype(int), 1
    )
    voxel_coords = np.minimum(
        np.floor((points - min_coords) / voxel_size).astype(int),
        grid_size - 1,
    )
    flat = np.ravel_multi_index(voxel_coords.T, grid_size)

    sort_idx = np.argsort(flat)
    unique_flat, first = np.unique(flat[sort_idx], return_index=True)
    counts = np.diff(np.append(first, len(flat)))
    down_pts = np.add.reduceat(points[sort_idx], first, axis=0) / counts[:, np.newaxis]

    # Map each original point directly to its voxel's index in down_pts.
    # Avoids a KDTree query — the voxel assignment is already known.
    voxel_to_down = np.empty(int(np.prod(grid_size)), dtype=np.intp)
    voxel_to_down[unique_flat] = np.arange(len(unique_flat))
    voxel_idx = voxel_to_down[flat]

    return down_pts, voxel_idx


def density_filter(
    points: np.ndarray, voxel_size: float, min_points: int
) -> np.ndarray:
    """Remove points whose voxel has fewer than min_points occupants.

    Args:
        points: (n, 3) numpy array.
        voxel_size: Edge length of each cubic voxel.
        min_points: Minimum number of points a voxel must contain to be kept.

    Returns:
        Filtered (m, 3) numpy array (m <= n).
    """
    if len(points) == 0:
        return points

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    grid_size = np.maximum(grid_size, 1)

    voxel_coords = np.floor((points - min_coords) / voxel_size).astype(int)
    voxel_coords = np.minimum(voxel_coords, grid_size - 1)

    indices = np.ravel_multi_index(voxel_coords.T, grid_size)
    voxel_counts = np.bincount(indices, minlength=int(np.prod(grid_size)))
    mask = voxel_counts[indices] >= min_points

    return points[mask]
