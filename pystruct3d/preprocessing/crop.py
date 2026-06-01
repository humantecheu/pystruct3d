import numpy as np


def _mask_boundaries(
    mask: np.ndarray,
    margin: int = 0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = np.nonzero(mask)
    h, w = mask.shape
    return (
        max(0, int(np.min(x)) - margin),
        max(0, int(np.min(y)) - margin),
    ), (
        min(h, int(np.max(x)) + margin),
        min(w, int(np.max(y)) + margin),
    )


def crop_roi(
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    resolution_m: float = 0.1,
    threshold: float = 0.1,
    margin_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float], tuple[float, float]]:
    """Crop a point cloud to its populated XY region via a 2D density histogram.

    Builds a 2D occupancy histogram over XY, normalises it, and keeps only the
    points inside the bounding box of cells whose normalised occupancy exceeds
    ``threshold``, expanded by ``margin_m`` on all sides.

    Args:
        xyz: (N, 3) XYZ point array.
        rgb: (N, 3) colour array, or None.
        resolution_m: Histogram cell size in metres.
        threshold: Normalised occupancy threshold (0–1); cells below this are
            treated as empty.
        margin_m: Padding in metres added around the occupied region on each side.

    Returns:
        Tuple of ``(cropped_xyz, cropped_rgb, x_range, y_range)`` where
        ``x_range = (x_min, x_max)`` and ``y_range = (y_min, y_max)``.
    """
    min_x, min_y = np.min(xyz[:, :2], axis=0)
    max_x, max_y = np.max(xyz[:, :2], axis=0)

    d_min_x = min_x // resolution_m - 1 if min_x < 0 else min_x // resolution_m
    d_max_x = max_x // resolution_m if max_x < 0 else max_x // resolution_m + 1
    d_min_y = min_y // resolution_m - 1 if min_y < 0 else min_y // resolution_m
    d_max_y = max_y // resolution_m if max_y < 0 else max_y // resolution_m + 1

    n_x_bins = int(d_max_x - d_min_x)
    n_y_bins = int(d_max_y - d_min_y)

    xy_hist, x_edges, y_edges = np.histogram2d(
        xyz[:, 0],
        xyz[:, 1],
        bins=(n_x_bins, n_y_bins),
        range=((d_min_x, d_max_x), (d_min_y, d_max_y)),
    )
    mask = (xy_hist / np.max(xy_hist)) > threshold

    bmin, bmax = _mask_boundaries(mask, int(margin_m // resolution_m))

    x_crop_min = float(x_edges[bmin[0]])
    x_crop_max = float(x_edges[bmax[0]])
    y_crop_min = float(y_edges[bmin[1]])
    y_crop_max = float(y_edges[bmax[1]])

    keep = (
        (x_crop_min <= xyz[:, 0])
        & (xyz[:, 0] <= x_crop_max)
        & (y_crop_min <= xyz[:, 1])
        & (xyz[:, 1] <= y_crop_max)
    )
    return (
        xyz[keep],
        rgb[keep] if rgb is not None else None,
        (x_crop_min, x_crop_max),
        (y_crop_min, y_crop_max),
    )
