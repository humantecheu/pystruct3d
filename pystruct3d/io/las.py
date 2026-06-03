import laspy
import numpy as np
from pathlib import Path


def read_las_file(las_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a LAS / LAZ file.

    Args:
        las_path: path to a .las or .laz file.

    Returns:
        Tuple of (xyz, rgb) arrays shaped (N, 3). RGB is normalised to [0, 1].
    """
    if isinstance(las_path, str):
        las_path = Path(las_path)

    if las_path.suffix not in {".las", ".laz"}:
        raise ValueError(f"File format '{las_path.suffix}' must be '.las' or '.laz'.")

    las_file = laspy.read(las_path)

    xyz = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    try:
        rgb = np.vstack((las_file.red, las_file.green, las_file.blue)).transpose()
        if np.max(rgb) > np.iinfo(np.uint8).max:
            rgb = rgb / np.iinfo(np.uint16).max
        elif np.max(rgb) > 1:
            rgb = rgb / np.iinfo(np.uint8).max
    except AttributeError:
        rgb = np.zeros(xyz.shape)

    return xyz, rgb.astype(np.float64)


def write_las_file(las_path: Path | str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a LAS file from XYZ and RGB numpy arrays.

    Args:
        las_path: output .las / .laz path.
        xyz: point coordinates, shape (N, 3).
        rgb: normalised RGB colours in [0, 1], shape (N, 3).
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    las.red = (rgb[:, 0] * 255).astype(int)
    las.green = (rgb[:, 1] * 255).astype(int)
    las.blue = (rgb[:, 2] * 255).astype(int)

    las.write(str(las_path))
