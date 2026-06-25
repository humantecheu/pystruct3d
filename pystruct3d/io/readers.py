# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from pystruct3d.io.e57 import read_e57_file
from pystruct3d.io.las import read_las_file

logger = logging.getLogger(__name__)

_LAS_EXTENSIONS = {".las", ".laz"}
_E57_EXTENSIONS = {".e57"}
_OPEN3D_EXTENSIONS = {".pcd", ".ply", ".xyz", ".xyzn", ".xyzrgb", ".pts"}


def _read_open3d_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a point cloud via open3d (PCD, PLY, XYZ, PTS, …)."""
    import open3d as o3d

    t0 = time.perf_counter()
    logger.info("Reading %s via open3d", path.name)
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    logger.info(
        "Loaded %d points from %s in %.1fs",
        len(xyz),
        path.name,
        time.perf_counter() - t0,
    )
    return xyz, np.asarray(pcd.colors)


def read_point_cloud(
    path: Path | str,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a point cloud file, routing to the correct reader by extension.

    Supported formats:
        - LAS / LAZ  (.las, .laz)               via laspy
        - E57        (.e57)                      via pye57
        - PCD / PLY / XYZ / PTS  (and others)   via open3d

    Args:
        path: path to the point cloud file.
        progress: show a tqdm progress bar while loading (LAS and E57 only).
            Defaults to True.

    Returns:
        Tuple of (xyz, rgb) arrays shaped (N, 3). RGB is normalised to [0, 1].
    """
    if isinstance(path, str):
        path = Path(path)

    ext = path.suffix.lower()

    if ext in _LAS_EXTENSIONS:
        return read_las_file(path, progress=progress)
    if ext in _E57_EXTENSIONS:
        return read_e57_file(path, progress=progress)
    if ext in _OPEN3D_EXTENSIONS:
        return _read_open3d_file(path)

    raise ValueError(
        f"Unsupported point cloud format '{ext}'. "
        f"Supported: {sorted(_LAS_EXTENSIONS | _E57_EXTENSIONS | _OPEN3D_EXTENSIONS)}"
    )
