# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import laspy
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 10_000_000  # points per chunk for streaming reads


def read_las_file(
    las_path: Path | str,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a LAS / LAZ file.

    Args:
        las_path: path to a .las or .laz file.
        progress: show a tqdm progress bar while loading. Defaults to True.

    Returns:
        Tuple of (xyz, rgb) arrays shaped (N, 3). RGB is normalised to [0, 1].
    """
    if isinstance(las_path, str):
        las_path = Path(las_path)

    if las_path.suffix not in {".las", ".laz"}:
        raise ValueError(f"File format '{las_path.suffix}' must be '.las' or '.laz'.")

    t0 = time.perf_counter()
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []

    with laspy.open(las_path) as reader:
        total = reader.header.point_count
        has_rgb = "red" in reader.header.point_format.standard_dimension_names
        n_chunks = math.ceil(total / _CHUNK_SIZE) if total > 0 else 1
        logger.info("Reading %s: %d points, rgb=%s", las_path.name, total, has_rgb)

        for chunk in tqdm(
            reader.chunk_iterator(_CHUNK_SIZE),
            total=n_chunks,
            desc=las_path.name,
            unit="chunk",
            disable=not progress,
        ):
            xyz_parts.append(np.vstack((chunk.x, chunk.y, chunk.z)).T)
            if has_rgb:
                rgb_parts.append(np.vstack((chunk.red, chunk.green, chunk.blue)).T)

    xyz = np.vstack(xyz_parts)

    if has_rgb:
        rgb = np.vstack(rgb_parts).astype(np.float64)
        if np.max(rgb) > np.iinfo(np.uint8).max:
            rgb /= np.iinfo(np.uint16).max
        elif np.max(rgb) > 1:
            rgb /= np.iinfo(np.uint8).max
    else:
        rgb = np.zeros(xyz.shape, dtype=np.float64)

    logger.info(
        "Loaded %d points from %s in %.1fs",
        len(xyz),
        las_path.name,
        time.perf_counter() - t0,
    )
    return xyz, rgb


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
