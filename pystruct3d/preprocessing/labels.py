from __future__ import annotations

import numpy as np

# Tab20 palette — identical to matplotlib's "tab20" colormap, inlined to avoid
# pulling in matplotlib as a runtime dependency.
_TAB20 = np.array(
    [
        [0.12156863, 0.46666667, 0.70588235],
        [0.68235294, 0.78039216, 0.90980392],
        [1.00000000, 0.49803922, 0.05490196],
        [1.00000000, 0.73333333, 0.47058824],
        [0.17254902, 0.62745098, 0.17254902],
        [0.59607843, 0.87450980, 0.54117647],
        [0.83921569, 0.15294118, 0.15686275],
        [1.00000000, 0.59607843, 0.58823529],
        [0.58039216, 0.40392157, 0.74117647],
        [0.77254902, 0.69019608, 0.83529412],
        [0.54901961, 0.33725490, 0.29411765],
        [0.76862745, 0.61176471, 0.58039216],
        [0.89019608, 0.46666667, 0.76078431],
        [0.96862745, 0.71372549, 0.82352941],
        [0.49803922, 0.49803922, 0.49803922],
        [0.78039216, 0.78039216, 0.78039216],
        [0.73725490, 0.74117647, 0.13333333],
        [0.85882353, 0.85882353, 0.55294118],
        [0.09019608, 0.74509804, 0.81176471],
        [0.61960784, 0.85490196, 0.89803922],
    ],
    dtype=np.float64,
)


def filter_ids(
    points: np.ndarray,
    label_arr: np.ndarray,
    label_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a point cloud to a single semantic label.

    Args:
        points: Point array of shape (n, 3).
        label_arr: Integer label array of shape (n,).
        label_id: Label value to keep.

    Returns:
        Tuple of (indices, filtered_points) where indices has shape (k,) and
        filtered_points has shape (k, 3).
    """
    indices = np.where(label_arr == int(label_id))[0]
    return indices, points[indices]


def labels_to_color(labels: np.ndarray) -> np.ndarray:
    """Map an integer label array to RGB colors using the tab20 palette.

    Background points (label <= 0) are mapped to black. Positive labels cycle
    through the 20-color tab20 palette.

    Args:
        labels: Integer label array of shape (n,).

    Returns:
        Float RGB array of shape (n, 3) with values in [0, 1].
    """
    colors = _TAB20[labels % len(_TAB20)].copy()
    colors[labels <= 0] = 0.0
    return colors
