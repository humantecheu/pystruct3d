# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_e57_file(
    e57_path: Path | str,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read an E57 file, concatenating all scans into a single point cloud.

    Args:
        e57_path: path to a .e57 file.
        progress: show a tqdm progress bar while loading. Defaults to True.

    Returns:
        Tuple of (xyz, rgb) arrays shaped (N, 3). RGB is normalised to [0, 1].
    """
    import pye57

    if isinstance(e57_path, str):
        e57_path = Path(e57_path)

    if e57_path.suffix != ".e57":
        raise ValueError(f"File format '{e57_path.suffix}' must be '.e57'.")

    e57 = pye57.E57(str(e57_path))
    n_scans = e57.scan_count
    logger.info("Reading %s: %d scan(s)", e57_path.name, n_scans)

    t0 = time.perf_counter()
    xyz_parts, rgb_parts = [], []

    with tqdm(
        total=n_scans, desc=e57_path.name, unit="scan", disable=not progress
    ) as pbar:
        for i in range(n_scans):
            n_pts = e57.get_header(i).point_count
            pbar.set_postfix({"pts": f"{n_pts:,}"})
            logger.debug("Scan %d/%d: %d points", i + 1, n_scans, n_pts)

            t_scan = time.perf_counter()
            data = e57.read_scan(i, colors=True, ignore_missing_fields=True)

            xyz = np.column_stack((
                data["cartesianX"],
                data["cartesianY"],
                data["cartesianZ"],
            ))
            xyz_parts.append(xyz)

            if "colorRed" in data:
                rgb = np.column_stack((
                    data["colorRed"],
                    data["colorGreen"],
                    data["colorBlue"],
                )).astype(np.float64)
                if np.max(rgb) > np.iinfo(np.uint8).max:
                    rgb /= np.iinfo(np.uint16).max
                elif np.max(rgb) > 1:
                    rgb /= np.iinfo(np.uint8).max
            else:
                rgb = np.zeros(xyz.shape)
            rgb_parts.append(rgb)

            logger.debug(
                "  Scan %d/%d done: %d points in %.1fs",
                i + 1,
                n_scans,
                len(xyz),
                time.perf_counter() - t_scan,
            )
            pbar.update(1)

    xyz_all = np.vstack(xyz_parts)
    logger.info(
        "Loaded %d points from %s in %.1fs",
        len(xyz_all),
        e57_path.name,
        time.perf_counter() - t0,
    )
    return xyz_all, np.vstack(rgb_parts)
