#!/usr/bin/env python3
"""Evaluate structure predictions against CV4AEC ground truth.

Computes IoU statistics, vertex precision/recall, and centroid deviation for
each model/floor/class pair found in the GT directory, then matches the
corresponding prediction file by filename.

Expected JSON filename pattern (same for GT and pred directories):
    {model}_{floor}_{classname}.json
    classname must be one of: walls, columns, doors

Example layout:
    data/gt/
        05_MedOffice_01_F2_walls.json
        05_MedOffice_01_F2_columns.json
        05_MedOffice_01_F2_doors.json
    data/pred/
        05_MedOffice_01_F2_walls.json
        ...

Usage:
    uv run python scripts/evaluate_cv4aec.py data/gt/ data/pred/
    uv run python scripts/evaluate_cv4aec.py data/gt/ data/pred/ --model 05_MedOffice_01 --floor F2
    uv run python scripts/evaluate_cv4aec.py data/gt/ data/pred/ --visualize-3d
    uv run python scripts/evaluate_cv4aec.py data/gt/ data/pred/ --visualize-2d
"""

import argparse
import json
import re
import sys  # for sys.exit
from pathlib import Path

import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.io.cv4aec import bbox_from_cv4aec
from pystruct3d.metrics.bbox_iou import match_iou_stats
from pystruct3d.metrics.point_metric import centroid_deviation, vertex_precision_recall

_PATTERN = re.compile(
    r"^(?P<model>.+)_(?P<floor>[^_]+)_(?P<classname>walls|columns|doors)\.json$"
)

VERTEX_THRESHOLDS = (0.05, 0.10, 0.20)
CENTROID_RADIUS = 0.5  # metres


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load(path: Path) -> list[BBox]:
    with open(path) as f:
        elements = json.load(f)
    return [bbox_from_cv4aec(elem) for elem in elements]


def _discover(directory: Path) -> list[dict]:
    """Return a list of {model, floor, classname, path} for matching JSON files."""
    hits = []
    for p in sorted(directory.glob("*.json")):
        m = _PATTERN.match(p.name)
        if m:
            hits.append({**m.groupdict(), "path": p})
    return hits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> dict:
    if not pred_boxes:
        zero = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        vpr = {
            t: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
            for t in VERTEX_THRESHOLDS
        }
        return {"iou": zero, "vertex": vpr, "centroid_deviation": float("nan")}

    gt_c = np.stack([b.corner_points for b in gt_boxes])
    pd_c = np.stack([b.corner_points for b in pred_boxes])

    return {
        "iou": match_iou_stats(gt_c, pd_c),
        "vertex": vertex_precision_recall(gt_c, pd_c, thresholds=VERTEX_THRESHOLDS),
        "centroid_deviation": centroid_deviation(gt_boxes, pred_boxes, CENTROID_RADIUS),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print(label: str, n_gt: int, n_pred: int, results: dict) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {label}")
    print(f"  GT: {n_gt} structures   Pred: {n_pred} structures")
    print(bar)

    iou = results["iou"]
    print(
        f"  IoU (optimal LAP)   "
        f"min {iou['min']:.3f}  max {iou['max']:.3f}  "
        f"mean {iou['mean']:.3f}  median {iou['median']:.3f}  std {iou['std']:.3f}"
    )

    print(
        f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Matched':>8}"
    )
    for t in VERTEX_THRESHOLDS:
        v = results["vertex"][t]
        print(
            f"  {t:.2f} m        "
            f"{v['precision']:>10.3f}  {v['recall']:>8.3f}  "
            f"{v['f1']:>8.3f}  {v['matched']:>8}"
        )

    cd = results["centroid_deviation"]
    cd_str = (
        f"{cd:.3f} m" if not np.isnan(cd) else f"N/A (none within {CENTROID_RADIUS} m)"
    )
    print(f"\n  Centroid deviation  {cd_str}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _visualize_3d(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> None:
    from pystruct3d.visualization import Visualizer

    Visualizer().add_bbox(gt_boxes, color=[0.15, 0.45, 0.90]).add_bbox(
        pred_boxes, color=[0.90, 0.20, 0.10]
    ).show()


def _visualize_2d(
    gt_boxes: list[BBox], pred_boxes: list[BBox], title: str = ""
) -> None:
    """Floor-plan (XY) view of box footprints — GT in blue, prediction in red dashed."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 10))

    for box in gt_boxes:
        xy = box.corner_points[:4, :2]
        ax.add_patch(
            Polygon(xy, closed=True, fill=False, edgecolor="steelblue", linewidth=1.5)
        )

    for box in pred_boxes:
        xy = box.corner_points[:4, :2]
        ax.add_patch(
            Polygon(
                xy,
                closed=True,
                fill=False,
                edgecolor="tomato",
                linewidth=1.5,
                linestyle="--",
            )
        )

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title or "GT vs Prediction (floor plan)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(
        handles=[
            mpatches.Patch(color="steelblue", label="Ground Truth"),
            mpatches.Patch(color="tomato", label="Prediction"),
        ]
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CV4AEC structure predictions against ground truth."
    )
    parser.add_argument("gt_dir", type=Path, help="Directory containing GT JSON files")
    parser.add_argument(
        "pred_dir", type=Path, help="Directory containing prediction JSON files"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to a specific model (e.g. 05_MedOffice_01)",
    )
    parser.add_argument(
        "--floor", default=None, help="Filter to a specific floor (e.g. F2)"
    )
    parser.add_argument(
        "--class",
        dest="classname",
        default=None,
        choices=["walls", "columns", "doors"],
        help="Filter to one element class",
    )
    parser.add_argument(
        "--visualize-3d", action="store_true", help="Show 3D Open3D view per pair"
    )
    parser.add_argument(
        "--visualize-2d", action="store_true", help="Show 2D floor-plan view per pair"
    )
    args = parser.parse_args()

    gt_entries = _discover(args.gt_dir)
    if not gt_entries:
        sys.exit(f"No matching JSON files found in {args.gt_dir}")

    # Apply filters
    if args.model:
        gt_entries = [e for e in gt_entries if e["model"] == args.model]
    if args.floor:
        gt_entries = [e for e in gt_entries if e["floor"] == args.floor]
    if args.classname:
        gt_entries = [e for e in gt_entries if e["classname"] == args.classname]

    if not gt_entries:
        sys.exit("No GT entries remain after filtering.")

    missing = 0
    for entry in gt_entries:
        pred_path = args.pred_dir / entry["path"].name
        label = f"{entry['model']} / {entry['floor']} / {entry['classname']}"

        if not pred_path.exists():
            print(f"\n[SKIP] {label} — prediction file not found: {pred_path}")
            missing += 1
            continue

        gt_boxes = _load(entry["path"])
        pred_boxes = _load(pred_path)

        results = _evaluate(gt_boxes, pred_boxes)
        _print(label, len(gt_boxes), len(pred_boxes), results)

        if args.visualize_3d:
            _visualize_3d(gt_boxes, pred_boxes)
        if args.visualize_2d:
            _visualize_2d(gt_boxes, pred_boxes, title=label)

    if missing:
        print(f"\nWarning: {missing} GT file(s) had no matching prediction file.")


if __name__ == "__main__":
    main()
