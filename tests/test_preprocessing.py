"""Tests for pystruct3d.preprocessing — pca, crop, voxel."""

import numpy as np

from pystruct3d.preprocessing import (
    crop_roi,
    density_filter,
    downsample,
    rotate_by_pca,
    simple_pca,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_cloud(n: int = 300, seed: int = 0) -> np.ndarray:
    """Elongated cloud: X spread 10 m, Y spread 2 m, Z noise only."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0, 10, n),
        rng.uniform(0, 2, n),
        rng.normal(0, 0.01, n),
    ])


def _grid_cloud(
    x_range: tuple[float, float] = (0, 5),
    y_range: tuple[float, float] = (0, 3),
    n: int = 500,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(*x_range, n),
        rng.uniform(*y_range, n),
        np.zeros(n),
    ])


# ---------------------------------------------------------------------------
# simple_pca
# ---------------------------------------------------------------------------


def test_simple_pca_shapes():
    pts = _flat_cloud()
    mean, eigenvalues, eigenvectors = simple_pca(pts)
    assert mean.shape == (3,)
    assert eigenvalues.shape == (3,)
    assert eigenvectors.shape == (3, 3)


def test_simple_pca_mean():
    pts = _flat_cloud()
    mean, _, _ = simple_pca(pts)
    np.testing.assert_allclose(mean, pts.mean(axis=0), rtol=1e-5)


def test_simple_pca_eigenvalue_descending():
    pts = _flat_cloud()
    _, eigenvalues, _ = simple_pca(pts)
    assert eigenvalues[0] >= eigenvalues[1] >= eigenvalues[2]


def test_simple_pca_first_axis_along_x():
    """For a cloud elongated along X the first principal axis should be ~X."""
    pts = _flat_cloud()
    _, _, eigenvectors = simple_pca(pts)
    dominant = eigenvectors[:, 0]
    assert abs(dominant[0]) > 0.9


# ---------------------------------------------------------------------------
# rotate_by_pca
# ---------------------------------------------------------------------------


def test_rotate_by_pca_output_shape():
    pts = _flat_cloud()
    mean, _, evecs = simple_pca(pts)
    rotated = rotate_by_pca(pts, mean, evecs)
    assert rotated.shape == pts.shape


def test_rotate_by_pca_reduces_off_diagonal_covariance():
    """After PCA rotation, the covariance off-diagonals should be near zero."""
    pts = _flat_cloud()
    mean, _, evecs = simple_pca(pts)
    rotated = rotate_by_pca(pts, mean, evecs)
    cov = np.cov(rotated.T)
    off_diag = np.abs(cov - np.diag(np.diag(cov)))
    assert off_diag.max() < 0.5 * np.diag(cov).max()


# ---------------------------------------------------------------------------
# crop_roi
# ---------------------------------------------------------------------------


def test_crop_roi_reduces_point_count():
    xyz = _grid_cloud(x_range=(-10, 10), y_range=(-10, 10), n=1000)
    cropped, _, _, _ = crop_roi(xyz, resolution_m=0.5, threshold=0.05, margin_m=0.5)
    assert cropped.shape[0] <= xyz.shape[0]


def test_crop_roi_returns_subset_of_input():
    xyz = _grid_cloud(n=200)
    cropped, _, x_range, y_range = crop_roi(xyz, resolution_m=0.2, threshold=0.05)
    assert cropped.shape[1] == 3
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]


def test_crop_roi_passes_rgb():
    rng = np.random.default_rng(2)
    xyz = _grid_cloud(n=100)
    rgb = rng.uniform(0, 1, (100, 3))
    _, cropped_rgb, _, _ = crop_roi(xyz, rgb=rgb)
    assert cropped_rgb is not None
    assert cropped_rgb.shape[1] == 3
    assert cropped_rgb.shape[0] == crop_roi(xyz)[0].shape[0]


def test_crop_roi_no_rgb_returns_none():
    xyz = _grid_cloud(n=100)
    _, rgb_out, _, _ = crop_roi(xyz, rgb=None)
    assert rgb_out is None


def test_crop_roi_empty_histogram_guard():
    """A single-point cloud should not raise ZeroDivisionError."""
    xyz = np.array([[0.0, 0.0, 0.0]])
    cropped, _, _, _ = crop_roi(xyz, resolution_m=0.1, threshold=0.5)
    assert cropped.shape[1] == 3


# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


def test_downsample_reduces_point_count():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 10, (1000, 3))
    down, idx = downsample(pts, voxel_size=1.0)
    assert down.shape[0] < pts.shape[0]
    assert idx.shape == (1000,)


def test_downsample_voxel_idx_in_range():
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 5, (500, 3))
    down, idx = downsample(pts, voxel_size=0.5)
    assert idx.min() >= 0
    assert idx.max() < down.shape[0]


def test_downsample_empty_input():
    pts = np.empty((0, 3))
    down, idx = downsample(pts, voxel_size=0.5)
    assert down.shape[0] == 0
    assert idx.shape[0] == 0


def test_downsample_single_voxel():
    """All points in a tiny cloud in one voxel → one downsampled point."""
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    down, idx = downsample(pts, voxel_size=1.0)
    assert down.shape[0] == 1
    assert np.all(idx == 0)


# ---------------------------------------------------------------------------
# density_filter
# ---------------------------------------------------------------------------


def test_density_filter_removes_sparse_voxels():
    rng = np.random.default_rng(5)
    dense = rng.uniform(0, 1, (300, 3))
    sparse = rng.uniform(10, 11, (2, 3))
    pts = np.vstack([dense, sparse])
    filtered = density_filter(pts, voxel_size=2.0, min_points=5)
    # sparse region (2 points) should be removed; dense region kept
    assert filtered.shape[0] == dense.shape[0]


def test_density_filter_keeps_all_when_threshold_1():
    rng = np.random.default_rng(6)
    pts = rng.uniform(0, 5, (200, 3))
    filtered = density_filter(pts, voxel_size=0.5, min_points=1)
    assert filtered.shape[0] == pts.shape[0]


def test_density_filter_empty_input():
    pts = np.empty((0, 3))
    filtered = density_filter(pts, voxel_size=0.5, min_points=3)
    assert filtered.shape[0] == 0
