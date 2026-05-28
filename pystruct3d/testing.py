"""Synthetic data generators for testing, benchmarking, and quick demos."""

import numpy as np

from pystruct3d.bbox.bbox import BBox


def generate_bounding_boxes(
    n_boxes: int = 15,
    l_range: tuple[float, float] = (3, 5),
    w_range: tuple[float, float] = (1, 2),
    h_range: tuple[float, float] = (2, 3),
    rng: np.random.Generator | None = None,
) -> list[BBox]:
    """Generate axis-aligned bounding boxes tiled on a 2D grid with random sizes.

    Args:
        n_boxes: number of boxes to generate. Defaults to 15.
        l_range: length range in metres. Defaults to (3, 5).
        w_range: width range in metres. Defaults to (1, 2).
        h_range: height range in metres. Defaults to (2, 3).
        rng: random generator. Defaults to a fixed seed for reproducibility.

    Returns:
        Generated bounding boxes (may be fewer than n_boxes if the grid is exhausted).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    assert n_boxes > 1, "n_boxes must be greater than 1"
    x_range = (0, n_boxes)
    y_range = (0, n_boxes)
    z_range = (0.0, 2.0 * float(np.max(h_range)))

    boxes: list[BBox] = []
    x, y = 0.0, 0.0
    max_w = 0.0
    for _ in range(n_boxes):
        length = float(rng.uniform(*l_range))
        width = float(rng.uniform(*w_range))
        height = float(rng.uniform(*h_range))
        z = float(rng.uniform(*z_range))

        boxes.append(
            BBox.from_params(
                np.array([x, y, z]), (length, width, height), origin="corner"
            )
        )

        x += length + 1.0 + float(rng.uniform(0, 1))
        if x > x_range[1]:
            x = 0.0
            y += max_w + 1.0 + float(rng.uniform(0, 1))
        max_w = max(max_w, width)

        if y > y_range[1]:
            break

    return boxes


def shift_bounding_boxes(
    bounding_boxes: list[BBox],
    n_boxes_to_shift: int,
    shift_range: tuple[float, float] = (-1.0, 1.0),
    rng: np.random.Generator | None = None,
) -> list[BBox]:
    """Randomly select a subset of bounding boxes and apply a uniform random shift.

    Args:
        bounding_boxes: source bounding boxes to sample from.
        n_boxes_to_shift: number of boxes to select and shift.
        shift_range: per-axis shift range in metres. Defaults to (-1, 1).
        rng: random generator. Defaults to a fixed seed for reproducibility.

    Returns:
        Shifted copies of the selected boxes.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    indices = rng.choice(len(bounding_boxes), size=int(n_boxes_to_shift), replace=False)
    return [
        BBox(bounding_boxes[i].as_np_array() + rng.uniform(*shift_range, size=3))
        for i in indices
    ]


def create_bbox_lists() -> tuple[list[BBox], list[BBox]]:
    """Create a paired (ground-truth, predicted) set of bounding boxes for demos and benchmarks.

    Returns:
        gt_boxes: 20 synthetic ground-truth bounding boxes.
        pd_boxes: 80 % of gt boxes shifted by a small random offset.
    """
    rng = np.random.default_rng(0)
    gt_boxes = generate_bounding_boxes(20, rng=rng)
    n_to_shift = int(np.ceil(len(gt_boxes) * 0.8))
    pd_boxes = shift_bounding_boxes(gt_boxes, n_to_shift, rng=rng)
    return gt_boxes, pd_boxes
