"""Visual test: load a LAZ file, show before and after axis alignment with coordinate frame.

Usage:
    uv run python scripts/visualize_alignment.py /path/to/cloud.laz
"""

import sys

import numpy as np
import open3d as o3d

from pystruct3d.preprocessing.alignment import align_to_axes
from pystruct3d.utils.las_utils import read_las_file
from pystruct3d.visualization.visualization import Visualization

COORD_FRAME_SIZE = 5.0


def show(points: np.ndarray, colors: np.ndarray, title: str) -> None:
    v = Visualization()
    v.point_cloud_geometry(points, colors=colors)
    v.visu_list.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=COORD_FRAME_SIZE)
    )
    v.visualize(window_name=title)


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
