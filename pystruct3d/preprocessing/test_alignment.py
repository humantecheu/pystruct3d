"""Visual test: load combined_1cm.laz, show before and after axis alignment with coordinate frame."""

from pathlib import Path

import numpy as np
import open3d as o3d

from pystruct3d.preprocessing.alignment import align_to_axes
from pystruct3d.utils.las_utils import read_las_file
from pystruct3d.visualization.visualization import Visualization

DATA = Path(__file__).resolve().parents[4] / "data" / "combined_1cm.laz"

COORD_FRAME_SIZE = 5.0  # metres — scale to match the building


def show(points, colors, title):
    v = Visualization()
    v.point_cloud_geometry(points, colors=colors)
    v.visu_list.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=COORD_FRAME_SIZE)
    )
    v.visualize(window_name=title)


def main():
    pcd = read_las_file(str(DATA))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # centre on origin so the coordinate frame sits at the building centre
    centroid = points.mean(axis=0)
    centred = points - centroid

    rotated, R = align_to_axes(centred)
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print(f"Applied rotation: {angle:.2f} deg")

    show(centred, colors, "Before alignment")
    show(rotated, colors, "After alignment")


if __name__ == "__main__":
    main()
