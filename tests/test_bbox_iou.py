# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
"""Tests for pystruct3d.metrics.bbox_iou and pystruct3d.metrics.point_metric.

Covers iou_batch, match_iou_stats, vertex_precision_recall, plus the private
polygon helpers that underpin iou_batch.

Corner layout throughout: corners[0:4] = bottom face (lower z),
corners[4:8] = top face (upper z), matching BBox.order_points() convention.
"""

import numpy as np
import pytest

from pystruct3d.metrics.bbox_iou import (
    _poly_area,
    _polygon_clip,
    _to_ccw,
    iou_batch,
    match_iou_stats,
)
from pystruct3d.metrics.point_metric import vertex_precision_recall

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _axis_box(
    x: float, y: float, z: float, dx: float, dy: float, dz: float
) -> np.ndarray:
    """Return an (8, 3) axis-aligned box with given origin and extents."""
    return np.array(
        [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def unit_box() -> np.ndarray:
    """1×1×2 box at the origin, shape (8, 3)."""
    return _axis_box(0, 0, 0, 1, 1, 2)


@pytest.fixture
def gt1(unit_box) -> np.ndarray:
    """Single ground-truth box, shape (1, 8, 3)."""
    return unit_box[None]


@pytest.fixture
def pred_identical(unit_box) -> np.ndarray:
    return unit_box[None]


@pytest.fixture
def pred_no_overlap_x(unit_box) -> np.ndarray:
    """Shifted 5 units along X — no overlap."""
    shifted = unit_box.copy()
    shifted[:, 0] += 5.0
    return shifted[None]


@pytest.fixture
def pred_no_overlap_z(unit_box) -> np.ndarray:
    """Same XY footprint but at a different Z level — no volumetric overlap."""
    shifted = unit_box.copy()
    shifted[:, 2] += 10.0
    return shifted[None]


@pytest.fixture
def pred_half_x(unit_box) -> np.ndarray:
    """Shifted 0.5 along X: 0.5×1×2 = 1 intersection, volumes 2+2=4, union=3."""
    shifted = unit_box.copy()
    shifted[:, 0] += 0.5
    return shifted[None]


# ---------------------------------------------------------------------------
# _polygon_clip
# ---------------------------------------------------------------------------


class TestPolygonClip:
    def test_full_overlap(self):
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = _polygon_clip(square, square)
        assert result is not None
        assert len(result) >= 4

    def test_no_overlap(self):
        sq1 = [[0, 0], [1, 0], [1, 1], [0, 1]]
        sq2 = [[5, 5], [6, 5], [6, 6], [5, 6]]
        result = _polygon_clip(sq1, sq2)
        assert result is None

    def test_partial_overlap(self):
        sq1 = [[0, 0], [2, 0], [2, 2], [0, 2]]
        sq2 = [[1, 0], [3, 0], [3, 2], [1, 2]]
        result = _polygon_clip(sq1, sq2)
        assert result is not None
        xs = [p[0] for p in result]
        assert all(1.0 <= x <= 2.0 for x in xs)


# ---------------------------------------------------------------------------
# _poly_area
# ---------------------------------------------------------------------------


class TestPolyArea:
    def test_unit_square(self):
        x = np.array([[0.0, 1.0, 1.0, 0.0]])
        y = np.array([[0.0, 0.0, 1.0, 1.0]])
        assert np.isclose(_poly_area(x, y), 1.0)

    def test_rectangle(self):
        x = np.array([[0.0, 2.0, 2.0, 0.0]])
        y = np.array([[0.0, 0.0, 3.0, 3.0]])
        assert np.isclose(_poly_area(x, y), 6.0)

    def test_batch(self):
        # Two unit squares stacked
        x = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
        y = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        areas = _poly_area(x, y)
        assert areas.shape == (2,)
        np.testing.assert_allclose(areas, [1.0, 1.0])


# ---------------------------------------------------------------------------
# _to_ccw
# ---------------------------------------------------------------------------


class TestToCcw:
    def test_ccw_unchanged(self):
        faces = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=float)
        result = _to_ccw(faces.copy())
        np.testing.assert_array_equal(result, faces)

    def test_cw_reversed(self):
        cw = np.array([[[0, 0], [0, 1], [1, 1], [1, 0]]], dtype=float)
        result = _to_ccw(cw.copy())
        # Reversed should be CCW
        x, y = result[0, :, 0], result[0, :, 1]
        signed_area = np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
        assert signed_area <= 0  # CCW is negative signed area in standard screen coords

    def test_mixed_batch(self):
        ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cw = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        faces = np.stack([ccw, cw])
        result = _to_ccw(faces)

        # Both should now have the same winding
        def signed(f):
            x, y = f[:, 0], f[:, 1]
            return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))

        assert np.sign(signed(result[0])) == np.sign(signed(result[1]))


# ---------------------------------------------------------------------------
# iou_batch
# ---------------------------------------------------------------------------


class TestIouBatch:
    def test_identical_is_one(self, gt1, pred_identical):
        iou = iou_batch(gt1, pred_identical)
        assert iou.shape == (1, 1)
        assert np.isclose(iou[0, 0], 1.0, atol=1e-5)

    def test_no_overlap_x_is_zero(self, gt1, pred_no_overlap_x):
        iou = iou_batch(gt1, pred_no_overlap_x)
        assert np.isclose(iou[0, 0], 0.0, atol=1e-6)

    def test_no_overlap_z_is_zero(self, gt1, pred_no_overlap_z):
        iou = iou_batch(gt1, pred_no_overlap_z)
        assert np.isclose(iou[0, 0], 0.0, atol=1e-6)

    def test_half_overlap_value(self, gt1, pred_half_x):
        # intersection vol = 0.5×1×2 = 1, each box vol = 2, union = 3
        iou = iou_batch(gt1, pred_half_x)
        assert np.isclose(iou[0, 0], 1.0 / 3.0, atol=1e-5)

    def test_output_shape(self, unit_box):
        gt = np.stack([unit_box, unit_box + 3.0])  # (2, 8, 3)
        pred = np.stack([unit_box, unit_box + 1.0, unit_box + 10.0])  # (3, 8, 3)
        iou = iou_batch(gt, pred)
        assert iou.shape == (2, 3)

    def test_dtype_is_float32(self, gt1, pred_identical):
        iou = iou_batch(gt1, pred_identical)
        assert iou.dtype == np.float32

    def test_values_in_unit_interval(self):
        # Use properly structured boxes (position offsets, not per-corner noise)
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 5, size=(5, 3))
        positions[:, 2] = 0.0  # keep z-base at 0
        boxes = np.stack([_axis_box(*p, 1, 1, 2) for p in positions])
        iou = iou_batch(boxes, boxes)
        assert np.all(iou >= 0.0)
        assert np.all(iou <= 1.0)

    def test_symmetric(self, unit_box):
        box_a = unit_box[None]
        box_b = (unit_box + np.array([0.3, 0.2, 0.0]))[None]
        iou_ab = iou_batch(box_a, box_b)
        iou_ba = iou_batch(box_b, box_a)
        np.testing.assert_allclose(iou_ab, iou_ba.T, atol=1e-5)

    def test_diagonal_is_one_for_identical_batch(self, unit_box):
        # n copies of the same box — diagonal should all be 1.0
        n = 4
        boxes = np.stack([unit_box + i * 20 for i in range(n)])
        iou = iou_batch(boxes, boxes)
        np.testing.assert_allclose(np.diag(iou), np.ones(n), atol=1e-5)

    def test_contained_box_iou(self, unit_box):
        # A box contained within a larger box has IoU = small_vol / large_vol
        small = unit_box[None]  # 1×1×2 = vol 2
        large = _axis_box(0, 0, 0, 2, 2, 2)[None]  # 2×2×2 = vol 8
        iou = iou_batch(small, large)
        # intersection = small vol = 2, union = 8, IoU = 2/8 = 0.25
        assert np.isclose(iou[0, 0], 0.25, atol=1e-4)

    def test_quarter_overlap(self, unit_box):
        # 0.25×1×2 intersection
        shifted = unit_box.copy()
        shifted[:, 0] += 0.75  # overlap width = 0.25
        iou = iou_batch(unit_box[None], shifted[None])
        # inter_vol = 0.25×1×2 = 0.5, vol_gt = vol_pred = 2, union = 3.5
        expected = 0.5 / 3.5
        assert np.isclose(iou[0, 0], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# vertex_precision_recall
# ---------------------------------------------------------------------------


class TestVertexPrecisionRecall:
    def test_identical_is_perfect(self, gt1, pred_identical):
        res = vertex_precision_recall(gt1, pred_identical, thresholds=(0.01,))
        assert np.isclose(res[0.01]["precision"], 1.0)
        assert np.isclose(res[0.01]["recall"], 1.0)
        assert np.isclose(res[0.01]["f1"], 1.0)
        assert res[0.01]["matched"] == 8

    def test_far_prediction_is_zero(self, gt1, pred_no_overlap_x):
        res = vertex_precision_recall(gt1, pred_no_overlap_x, thresholds=(0.01,))
        assert res[0.01]["precision"] == 0.0
        assert res[0.01]["recall"] == 0.0
        assert res[0.01]["f1"] == 0.0

    def test_empty_prediction(self, gt1):
        pred_empty = np.zeros((0, 8, 3))
        res = vertex_precision_recall(gt1, pred_empty, thresholds=(0.05,))
        assert res[0.05]["precision"] == 0.0
        assert res[0.05]["recall"] == 0.0
        assert res[0.05]["f1"] == 0.0

    def test_multiple_thresholds_returned(self, gt1, pred_half_x):
        thresholds = (0.05, 0.10, 0.50)
        res = vertex_precision_recall(gt1, pred_half_x, thresholds=thresholds)
        assert set(res.keys()) == set(thresholds)

    def test_looser_threshold_more_matches(self, gt1, pred_half_x):
        res = vertex_precision_recall(gt1, pred_half_x, thresholds=(0.05, 1.0))
        assert res[1.0]["matched"] >= res[0.05]["matched"]

    def test_precision_recall_keys(self, gt1, pred_identical):
        res = vertex_precision_recall(gt1, pred_identical, thresholds=(0.1,))
        assert {"precision", "recall", "f1", "matched"} == set(res[0.1].keys())

    def test_values_in_unit_interval(self, gt1, pred_half_x):
        res = vertex_precision_recall(gt1, pred_half_x, thresholds=(0.05, 0.5))
        for v in res.values():
            assert 0.0 <= v["precision"] <= 1.0
            assert 0.0 <= v["recall"] <= 1.0
            assert 0.0 <= v["f1"] <= 1.0

    def test_more_pred_than_gt_lowers_precision(self, unit_box):
        # 1 GT box, 4 pred boxes (one matches, three far away)
        gt = unit_box[None]
        extra = np.stack([
            unit_box,
            unit_box + np.array([10, 0, 0]),
            unit_box + np.array([20, 0, 0]),
            unit_box + np.array([30, 0, 0]),
        ])
        res = vertex_precision_recall(gt, extra, thresholds=(0.01,))
        # 8 GT verts, 32 pred verts, 8 matched (perfect GT→pred[0] assignment)
        assert np.isclose(res[0.01]["precision"], 8 / 32)
        assert np.isclose(res[0.01]["recall"], 1.0)

    def test_f1_is_harmonic_mean(self, gt1, pred_half_x):
        res = vertex_precision_recall(gt1, pred_half_x, thresholds=(0.6,))
        v = res[0.6]
        if v["precision"] + v["recall"] > 0:
            expected_f1 = (
                2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"])
            )
            assert np.isclose(v["f1"], expected_f1)


# ---------------------------------------------------------------------------
# match_iou_stats
# ---------------------------------------------------------------------------


class TestMatchIouStats:
    def test_identical_all_one(self, gt1, pred_identical):
        stats = match_iou_stats(gt1, pred_identical)
        assert np.isclose(stats["min"], 1.0, atol=1e-5)
        assert np.isclose(stats["max"], 1.0, atol=1e-5)
        assert np.isclose(stats["mean"], 1.0, atol=1e-5)
        assert np.isclose(stats["median"], 1.0, atol=1e-5)
        assert np.isclose(stats["std"], 0.0, atol=1e-5)

    def test_no_overlap_all_zero(self, gt1, pred_no_overlap_x):
        stats = match_iou_stats(gt1, pred_no_overlap_x)
        assert np.isclose(stats["mean"], 0.0, atol=1e-6)
        assert np.isclose(stats["max"], 0.0, atol=1e-6)

    def test_unmatched_gt_padded_with_zero(self, unit_box):
        # 3 GT boxes, 1 pred matching only the first GT box
        gt = np.stack([unit_box, unit_box + 20, unit_box + 40])  # (3,8,3)
        pred = unit_box[None]  # (1,8,3)
        stats = match_iou_stats(gt, pred)
        # Best assignment: pred[0] → gt[0] (IoU=1); gt[1], gt[2] → 0
        assert np.isclose(stats["max"], 1.0, atol=1e-5)
        assert np.isclose(stats["mean"], 1.0 / 3.0, atol=1e-5)
        assert np.isclose(stats["min"], 0.0, atol=1e-6)

    def test_keys_present(self, gt1, pred_identical):
        stats = match_iou_stats(gt1, pred_identical)
        assert {"min", "max", "mean", "median", "std"} == set(stats.keys())

    def test_values_are_float(self, gt1, pred_identical):
        stats = match_iou_stats(gt1, pred_identical)
        assert all(isinstance(v, float) for v in stats.values())

    def test_half_overlap_mean(self, gt1, pred_half_x):
        stats = match_iou_stats(gt1, pred_half_x)
        assert np.isclose(stats["mean"], 1.0 / 3.0, atol=1e-5)

    def test_batch_two_boxes(self, unit_box):
        # 2 GT, 2 pred: pred[0]↔gt[0] perfect, pred[1]↔gt[1] half-overlap
        gt = np.stack([unit_box, unit_box + np.array([5, 0, 0])])
        pred = np.stack([unit_box, unit_box + np.array([5.5, 0, 0])])
        stats = match_iou_stats(gt, pred)
        assert 0.0 < stats["mean"] <= 1.0
        assert stats["max"] >= stats["mean"] >= stats["min"]
