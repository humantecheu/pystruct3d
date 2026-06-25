"""Baseline benchmarks for pystruct3d hot paths.

Run with:
    uv run python scripts/benchmark.py

Each benchmark prints mean ± std over N_RUNS repetitions.
Re-run after any optimisation to detect regressions or measure gains.
"""

import timeit

import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.metrics.bbox_iou import mean_bbox_iou
from pystruct3d.metrics.volumetric_iou import voxelize_bbox
from pystruct3d.metrics.voxelization_limits import voxelization_limits
from pystruct3d.testing import create_bbox_lists

# ── shared fixtures ────────────────────────────────────────────────────────────

N_RUNS = 7
N_PTS_SMALL = 500_000
N_PTS_LARGE = 5_000_000
VOXEL_SIZE = 0.05

_rng = np.random.default_rng(42)

_WALL_PTS = np.array([
    [0.0, 0.0, 0.0],
    [5.0, 0.0, 0.0],
    [5.0, 0.2, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 3.0],
    [5.0, 0.0, 3.0],
    [5.0, 0.2, 3.0],
    [0.0, 0.2, 3.0],
])

BBOX = BBox(_WALL_PTS.copy())

SMALL_CLOUD = _rng.uniform(-6, 6, size=(N_PTS_SMALL, 3))
LARGE_CLOUD = _rng.uniform(-6, 6, size=(N_PTS_LARGE, 3))

GT_BOXES, PD_BOXES = create_bbox_lists()

VOL_LIMITS = voxelization_limits(GT_BOXES)


# ── helpers ────────────────────────────────────────────────────────────────────


def _run(label: str, stmt, setup=lambda: None, n_runs: int = N_RUNS) -> None:
    times = []
    for _ in range(n_runs):
        setup()
        t = timeit.timeit(stmt, number=1) * 1_000  # → ms
        times.append(t)
    arr = np.array(times)
    print(f"  {label:<52}  {arr.mean():8.1f} ± {arr.std():6.1f} ms")


# ── benchmarks ─────────────────────────────────────────────────────────────────


def bench_points_in_bbox() -> None:
    """OBB containment test via Open3D."""
    print("\npoints_in_bbox  [OrientedBoundingBox, open3d]")
    _run(
        f"small cloud  ({N_PTS_SMALL:,} pts)", lambda: BBOX.points_in_bbox(SMALL_CLOUD)
    )
    _run(
        f"large cloud  ({N_PTS_LARGE:,} pts)", lambda: BBOX.points_in_bbox(LARGE_CLOUD)
    )


def bench_points_in_bbox_2d() -> None:
    """2D footprint containment via plane-normal dot product (ignores Z)."""
    print("\npoints_in_bbox_2d  [dot-product, side planes only]")
    _run(
        f"small cloud  ({N_PTS_SMALL:,} pts)",
        lambda: BBOX.points_in_bbox_2d(SMALL_CLOUD),
    )
    _run(
        f"large cloud  ({N_PTS_LARGE:,} pts)",
        lambda: BBOX.points_in_bbox_2d(LARGE_CLOUD),
    )


def bench_points_in_bbox_soft() -> None:
    """Soft Gaussian containment (distance to surface, threshold=0.7)."""
    print("\npoints_in_bbox_soft  [Gaussian, threshold=0.7]")
    _run(
        f"small cloud  ({N_PTS_SMALL:,} pts)",
        lambda: BBOX.points_in_bbox_soft(SMALL_CLOUD, 0.7),
    )
    _run(
        f"large cloud  ({N_PTS_LARGE:,} pts)",
        lambda: BBOX.points_in_bbox_soft(LARGE_CLOUD, 0.7),
    )


def bench_fit_horizontal_aligned() -> None:
    """Minimum-volume OBB fitting via convex-hull edge enumeration."""
    print("\nfit_horizontal_aligned  [per-angle rotation loop]")
    for n in (1_000, 10_000, 100_000):
        pts = _rng.uniform(-5, 5, size=(n, 3))
        pts[:, 2] *= 0.1  # flat cloud — realistic wall shape
        _run(f"{n:>7,} pts", lambda p=pts: BBox().fit_horizontal_aligned(p))


def bench_voxelize_bbox() -> None:
    """Dense-meshgrid voxelization of a single bounding box."""
    print(f"\nvoxelize_bbox  [meshgrid, voxel_size={VOXEL_SIZE}]")
    limits = voxelization_limits(BBOX)
    _run(
        f"wall bbox 5×0.2×3 m  voxel={VOXEL_SIZE} m",
        lambda: voxelize_bbox(BBOX, limits, VOXEL_SIZE),
    )
    for vs in (0.1, 0.02):
        limits2 = voxelization_limits(BBOX)
        _run(
            f"wall bbox 5×0.2×3 m  voxel={vs} m",
            lambda lim=limits2, v=vs: voxelize_bbox(BBOX, lim, v),
        )


def bench_mean_bbox_iou() -> None:
    """Pairwise analytical bbox IoU + Hungarian assignment."""
    print(f"\nmean_bbox_iou  [{len(GT_BOXES)} gt × {len(PD_BOXES)} pd boxes]")
    _run("mean_bbox_iou", lambda: mean_bbox_iou(GT_BOXES, PD_BOXES))


def bench_order_points() -> None:
    """Corner-point reordering called after every mutating operation."""
    print("\nBBox.order_points  [called after every rotate/expand/etc.]")
    bbox_copy = BBox(_WALL_PTS.copy())
    _run(
        "order_points (axis-aligned wall)", lambda: bbox_copy.order_points(), n_runs=50
    )


# ── entry point ────────────────────────────────────────────────────────────────

BENCHMARKS = [
    bench_points_in_bbox,
    bench_points_in_bbox_2d,
    bench_points_in_bbox_soft,
    bench_fit_horizontal_aligned,
    bench_voxelize_bbox,
    bench_mean_bbox_iou,
    bench_order_points,
]

if __name__ == "__main__":
    print("=" * 72)
    print("pystruct3d  —  hot-path baseline benchmarks")
    print(f"{'runs per benchmark:':<30} {N_RUNS}")
    print("=" * 72)

    for bench in BENCHMARKS:
        bench()

    print("\n" + "=" * 72)
    print("Done. Save this output as the baseline before optimising.")
    print("=" * 72)
