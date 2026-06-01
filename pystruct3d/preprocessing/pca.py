import numpy as np


def simple_pca(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via eigendecomposition of the covariance matrix.

    Args:
        arr: (N, D) input array where D = 2 or 3.

    Returns:
        mean: (D,) centroid of the data.
        eigenvalues: (D,) eigenvalues in descending order.
        eigenvectors: (D, D) matrix whose *columns* are eigenvectors ordered
            by descending eigenvalue (largest principal axis is column 0).
    """
    mean = np.mean(arr, axis=0)
    cov = np.cov((arr - mean).T)
    eigval, eigvec = np.linalg.eigh(cov)
    idx = np.argsort(eigval)[::-1]
    return mean, eigval[idx], eigvec[:, idx]


def rotate_by_pca(
    point_cloud: np.ndarray,
    mean: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Rotate a point cloud so its two largest principal axes align with XY.

    Constructs a right-handed rotation matrix from the first two principal axes
    (columns 0 and 1 of ``eigenvectors``) and their cross product, then applies
    it to ``point_cloud - mean``.

    Args:
        point_cloud: (N, 3) XYZ array.
        mean: (3,) centroid — first output of :func:`simple_pca`.
        eigenvectors: (3, 3) eigenvector matrix — third output of
            :func:`simple_pca`.  Columns are principal axes in descending order.

    Returns:
        (N, 3) rotated point cloud, centred at the origin.
    """
    ev1, ev2 = eigenvectors[:, 0], eigenvectors[:, 1]
    R = np.stack([ev1, ev2, np.cross(ev1, ev2)], axis=0)
    return (point_cloud - mean) @ R.T
