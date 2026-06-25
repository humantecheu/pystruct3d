# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
"""Visual test: load a LAZ file, show before and after axis alignment with coordinate frame.

Usage:
    uv run python scripts/visualize_alignment.py /path/to/cloud.laz
"""

import sys

import numpy as np

from pystruct3d.io.las import read_las_file
from pystruct3d.preprocessing.alignment import align_to_axes
from pystruct3d.visualization import Visualizer

COORD_FRAME_SIZE = 5.0


def show(points: np.ndarray, colors: np.ndarray, title: str) -> None:
    Visualizer().add_points(points, colors=colors).add_coordinate_frame(
        size=COORD_FRAME_SIZE
    ).show(window_name=title)


def main(laz_path: str) -> None:
    pcd = read_las_file(laz_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    centroid = points.mean(axis=0)
    centred = points - centroid

    rotated, R = align_to_axes(centred)
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print(f"Applied rotation: {angle:.2f} deg")

    show(centred, colors, "Before alignment")
    show(rotated, colors, "After alignment")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/cloud.laz>")
        sys.exit(1)
    main(sys.argv[1])
