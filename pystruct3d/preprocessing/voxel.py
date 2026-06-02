import numpy as np


def _to_flat_voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Map each point to a flat voxel index in a local grid.

    Args:
        points: (n, 3) point array.
        voxel_size: Edge length of each cubic voxel.

    Returns:
        (n,) flat voxel index per point (row-major over the local grid).
    """
    min_coords = points.min(axis=0)
    grid_size = np.maximum(
        np.ceil((points.max(axis=0) - min_coords) / voxel_size).astype(int), 1
    )
    voxel_coords = np.minimum(
        np.floor((points - min_coords) / voxel_size).astype(int),
        grid_size - 1,
    )
    return np.ravel_multi_index(voxel_coords.T, grid_size)


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
    if points.shape[0] == 0:
        return points, np.empty(0, dtype=np.intp)

    flat = _to_flat_voxel_indices(points, voxel_size)

    sort_idx = np.argsort(flat)
    unique_flat, first = np.unique(flat[sort_idx], return_index=True)
    counts = np.diff(np.append(first, len(flat)))
    down_pts = np.add.reduceat(points[sort_idx], first, axis=0) / counts[:, np.newaxis]

    # searchsorted maps each flat voxel index to its rank in unique_flat in
    # O(n log m) — avoids allocating the full voxel grid.
    voxel_idx = np.searchsorted(unique_flat, flat)

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

    flat = _to_flat_voxel_indices(points, voxel_size)
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    mask = counts[inverse] >= min_points

    return points[mask]
