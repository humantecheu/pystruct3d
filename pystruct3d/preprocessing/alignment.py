import numpy as np


def _dominant_wall_angle(
    xy: np.ndarray,
    n_samples: int = 5000,
    n_pairs: int = 30000,
    min_dist: float = 0.5,
    n_bins: int = 720,
) -> float:
    """Estimate the dominant wall orientation from an XY point set.

    Randomly samples point pairs, keeps only those separated by at least
    min_dist (so pairs run along wall surfaces rather than across the thin
    scan-noise dimension), and builds a histogram of their edge angles folded
    to [0, π).  The histogram peak gives the dominant wall direction.

    Args:
        xy: (N, 2) array of XY coordinates.
        n_samples: Random subsample size for speed.
        n_pairs: Number of random point pairs to evaluate.
        min_dist: Minimum XY separation (metres) to accept a pair.
        n_bins: Histogram resolution over [0, π).

    Returns:
        Dominant wall angle in radians, in [0, π).
    """
    rng = np.random.default_rng(0)

    if len(xy) > n_samples:
        xy = xy[rng.choice(len(xy), n_samples, replace=False)]

    i = rng.integers(0, len(xy), n_pairs)
    j = rng.integers(0, len(xy), n_pairs)

    delta = xy[j] - xy[i]
    dist = np.hypot(delta[:, 0], delta[:, 1])

    mask = dist > min_dist
    # if the cloud is very small, relax the threshold
    if mask.sum() < 200:
        min_dist = np.percentile(dist, 10)
        mask = dist > min_dist

    angles = np.arctan2(delta[mask, 1], delta[mask, 0]) % np.pi

    hist, edges = np.histogram(angles, bins=n_bins, range=(0, np.pi))
    return edges[np.argmax(hist)]


def align_to_axes(
    points: np.ndarray,
    labels: np.ndarray | None = None,
    wall_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a point cloud so its walls align with the X/Y axes.

    Uses an angle histogram over long-range point pairs (distance > 0.5 m) to
    find the dominant wall direction.  Long-range pairs run along wall surfaces
    rather than across the thin scan-noise dimension, making the estimate
    robust to point-cloud density artefacts and outlier noise.

    If wall labels are provided, only those points are used to estimate the
    orientation; the rotation is applied to all points.

    Args:
        points: (N, 3) array of XYZ point coordinates.
        labels: (N,) integer label array, optional.
        wall_label: Label value for wall points.  If None, all points are used.

    Returns:
        rotated_points: (N, 3) array with the rotation applied.
        rotation_matrix: (3, 3) rotation matrix (around Z axis).
    """
    source = (
        points[labels == wall_label]
        if (wall_label is not None and labels is not None)
        else points
    )

    angle = _dominant_wall_angle(source[:, :2])

    c, s = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    rotated_points = (rotation_matrix @ points.T).T
    return rotated_points, rotation_matrix
