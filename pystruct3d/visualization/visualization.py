import numpy as np
import open3d as o3d
from pystruct3d.bbox.bbox import BBox


class Visualization:
    """Class for basic Visualization of one or multiple point clouds or bounding boxes"""

    def __init__(self) -> None:
        self.visu_list = []

    def o3d_point_cloud(self, point_cloud: o3d.geometry.PointCloud):
        self.visu_list.append(point_cloud)

    def point_cloud_geometry(
        self, points: np.ndarray, unique_color=None, colors=np.empty((0,))
    ) -> None:
        """create an open3d point cloud from points

        Args:
            points (np.ndarray): points, shape (n, 3)
        """
        # initialize the point cloud with the points
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if np.any(colors):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if unique_color is not None:
            pcd.paint_uniform_color(unique_color)
        # append to Visualization list
        self.visu_list.append(pcd)

    def bbox_geometry(self, bboxes: list[BBox], color=[0, 1, 0]):
        """Creates an open3d line set from a bounding box. Points should be ordered!

        Args:
            bounding_box (bbox): bbox object, or list of BBox objects
            color (list, optional): RGB color, list in range of 0, 1. Defaults to [0, 1, 0].
        """

        def visualize_bbox(bx: BBox) -> None:
            corner_points_array = bx.as_np_array()
            # Lines are represented as pairs of indices referencing the list of points (i.e., the corners of the box)
            # fmt:off
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
                [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
                ]
            # fmt:on
            colors = [color for i in range(len(lines))]
            # initialize line set with the corner points
            bounding_box = o3d.geometry.LineSet()
            bounding_box.points = o3d.utility.Vector3dVector(
                corner_points_array
            )  # Flatten the points
            bounding_box.lines = o3d.utility.Vector2iVector(lines)
            bounding_box.colors = o3d.utility.Vector3dVector(colors)
            # append to Visualization list
            self.visu_list.append(bounding_box)

        # handle exception so that it accepts Bbox objects and list[BBox]
        try:
            for box in bboxes:
                visualize_bbox(box)
        except TypeError:
            visualize_bbox(bboxes)

    def points_geometry(self, points, color=[1, 0.706, 0]):
        """visualize few points e.g., endpoints

        Args:
            points (np.ndarray): points, shape (n, 3)
        """

        for pt in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(pt)
            sphere.paint_uniform_color(color)
            self.visu_list.append(sphere)

    def clear(self):
        """Clear visualization list"""
        self.visu_list = []

    def visualize(
        self, window_name="pystruct3D Visualizer", w_width=2560, w_height=1440
    ):
        """visualize list of geometries"""
        # open3d coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.visu_list.append(coord_frame)
        if self.visu_list:
            o3d.visualization.draw_geometries(
                self.visu_list,
                window_name=window_name,
                width=w_width,
                height=w_height,
            )
        else:
            print("Empty visulization list, did you create geometries?")

    def visualize_with_animation(
        self,
        window_name="pystruct3D Visualizer",
        w_width=2560,
        w_height=1440,
        animation_trajectory="",
    ):
        """visualize list of geometries"""
        # open3d coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.visu_list.append(coord_frame)
        if self.visu_list:
            o3d.visualization.draw_geometries_with_custom_animation(
                self.visu_list,
                window_name=window_name,
                width=w_width,
                height=w_height,
                optional_view_trajectory_json_file=animation_trajectory,
            )
        else:
            print("Empty visulization list, did you create geometries?")
