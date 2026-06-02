"""Tests for pystruct3d.bbox.BBox core methods."""

import numpy as np
import pytest

from pystruct3d.bbox import BBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_cloud(
    length: float, width: float, height: float, n: int = 400, seed: int = 0
) -> np.ndarray:
    """Random points uniformly sampled inside an axis-aligned box."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0, length, n),
        rng.uniform(0, width, n),
        rng.uniform(0, height, n),
    ])


# ---------------------------------------------------------------------------
# BBox import / construction
# ---------------------------------------------------------------------------


def test_bbox_importable_from_subpackage():
    """from pystruct3d.bbox import BBox must work after adding __init__.py."""
    assert BBox is not None


def test_from_params_center():
    b = BBox.from_params(np.array([0.0, 0.0, 0.0]), (4.0, 2.0, 3.0))
    assert abs(b.length() - 4.0) < 1e-6
    assert abs(b.width() - 2.0) < 1e-6
    assert abs(b.height() - 3.0) < 1e-6


def test_from_params_corner():
    b = BBox.from_params(np.array([0.0, 0.0, 0.0]), (4.0, 2.0, 3.0), origin="corner")
    center = np.mean(b.corner_points, axis=0)
    np.testing.assert_allclose(center, [2.0, 1.0, 1.5], atol=1e-6)


def test_from_params_invalid_origin():
    with pytest.raises(ValueError, match="origin"):
        BBox.from_params(np.zeros(3), (1.0, 1.0, 1.0), origin="top")


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


def test_dimensions_length_ge_width():
    b = BBox.from_params(np.zeros(3), (5.0, 2.0, 3.0))
    assert b.length() >= b.width()


def test_height():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    assert abs(b.height() - 3.0) < 1e-6


def test_volume():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    assert abs(b.volume() - 24.0) < 1e-5


# ---------------------------------------------------------------------------
# fit_horizontal_aligned
# ---------------------------------------------------------------------------


def test_fit_horizontal_aligned_recovers_box():
    pts = _box_cloud(5.0, 2.0, 3.0, n=500)
    b = BBox()
    b.fit_horizontal_aligned(pts)
    assert abs(b.length() - 5.0) < 0.1
    assert abs(b.width() - 2.0) < 0.1
    assert abs(b.height() - 3.0) < 0.1


def test_fit_horizontal_aligned_empty_warns():
    b = BBox()
    with pytest.warns(UserWarning):
        b.fit_horizontal_aligned(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# points_in_bbox
# ---------------------------------------------------------------------------


def test_points_in_bbox_inside():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    pts_inside = np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 1.0]])
    _inliers, indices = b.points_in_bbox(pts_inside)
    assert len(indices) == 2


def test_points_in_bbox_outside():
    b = BBox.from_params(np.zeros(3), (2.0, 2.0, 2.0))
    pts_outside = np.array([[10.0, 10.0, 10.0], [-5.0, 0.0, 0.0]])
    _inliers, indices = b.points_in_bbox(pts_outside)
    assert len(indices) == 0


def test_points_in_bbox_mixed():
    b = BBox.from_params(np.zeros(3), (2.0, 2.0, 2.0))
    pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    _, indices = b.points_in_bbox(pts)
    assert len(indices) == 1


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


def test_iou_identical_boxes():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    assert abs(b.iou(b) - 1.0) < 1e-4


def test_iou_non_overlapping():
    b1 = BBox.from_params(np.array([0.0, 0.0, 0.0]), (2.0, 2.0, 2.0))
    b2 = BBox.from_params(np.array([10.0, 0.0, 0.0]), (2.0, 2.0, 2.0))
    assert b1.iou(b2) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# translate / expand
# ---------------------------------------------------------------------------


def test_translate_shifts_all_corners():
    b = BBox.from_params(np.zeros(3), (2.0, 2.0, 2.0))
    original_center = np.mean(b.corner_points, axis=0).copy()
    shift = np.array([1.0, 2.0, 3.0])
    b.translate(shift)
    new_center = np.mean(b.corner_points, axis=0)
    np.testing.assert_allclose(new_center, original_center + shift, atol=1e-6)


def test_expand_increases_volume():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    vol_before = b.volume()
    b.expand(0.5)
    assert b.volume() > vol_before


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------


def test_rotate_preserves_center():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    center_before = np.mean(b.corner_points, axis=0).copy()
    b.rotate(45.0)
    center_after = np.mean(b.corner_points, axis=0)
    np.testing.assert_allclose(center_after, center_before, atol=1e-6)


def test_rotate_360_returns_to_original():
    b = BBox.from_params(np.zeros(3), (4.0, 2.0, 3.0))
    original = b.corner_points.copy()
    b.rotate(360.0)
    np.testing.assert_allclose(b.corner_points, original, atol=1e-6)
