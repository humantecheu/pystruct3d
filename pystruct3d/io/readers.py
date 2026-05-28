import numpy as np
from pathlib import Path

from pystruct3d.io.las import read_las_file
from pystruct3d.io.e57 import read_e57_file

_LAS_EXTENSIONS = {".las", ".laz"}
_E57_EXTENSIONS = {".e57"}
_OPEN3D_EXTENSIONS = {".pcd", ".ply", ".xyz", ".xyzn", ".xyzrgb", ".pts"}


def _read_open3d_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def read_point_cloud(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a point cloud file, routing to the correct reader by extension.

    Supported formats:
        - LAS / LAZ  (.las, .laz)               via laspy
        - E57        (.e57)                      via pye57
        - PCD / PLY / XYZ / PTS  (and others)   via open3d

    Args:
        path: path to the point cloud file.

    Returns:
        Tuple of (xyz, rgb) arrays shaped (N, 3). RGB is normalised to [0, 1].
    """
    if isinstance(path, str):
        path = Path(path)

    ext = path.suffix.lower()

    if ext in _LAS_EXTENSIONS:
        return read_las_file(path)
    if ext in _E57_EXTENSIONS:
        return read_e57_file(path)
    if ext in _OPEN3D_EXTENSIONS:
        return _read_open3d_file(path)

    raise ValueError(
        f"Unsupported point cloud format '{ext}'. "
        f"Supported: {sorted(_LAS_EXTENSIONS | _E57_EXTENSIONS | _OPEN3D_EXTENSIONS)}"
    )
