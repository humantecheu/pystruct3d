from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import open3d as o3d

from pystruct3d.bbox.bbox import BBox

_Geometry = o3d.geometry.Geometry3D


class Visualizer:
    """Fluent builder for Open3D visualizations.

    Each ``add_*`` method returns ``self`` so calls can be chained::

        Visualizer()
            .add_bbox(gt_boxes, color=[1, 0, 0], name="gt")
            .add_points(cloud)
            .add_coordinate_frame()
            .show()
    """

    def __init__(self) -> None:
        self._geometries: list[tuple[str | None, _Geometry]] = []

    # ── add methods ───────────────────────────────────────────────────────────

    def add_bbox(
        self,
        bbox: BBox | Sequence[BBox],
        color: list[float] | None = None,
        *,
        name: str | None = None,
    ) -> Visualizer:
        """Add one or more bounding boxes as wire-frame line sets.

        Args:
            bbox: a single BBox or a sequence of BBox objects.
            color: RGB color in [0, 1]. Defaults to green.
            name: optional tag for later removal with :meth:`remove`.

        Returns:
            self
        """
        if color is None:
            color = [0, 1, 0]
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        line_colors = [color for _ in lines]

        boxes = [bbox] if isinstance(bbox, BBox) else list(bbox)
        for box in boxes:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(box.as_np_array())
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(line_colors)
            self._geometries.append((name, ls))
        return self

    def add_points(
        self,
        points: np.ndarray,
        color: list[float] | None = None,
        colors: np.ndarray | None = None,
        *,
        name: str | None = None,
    ) -> Visualizer:
        """Add a point cloud from a numpy array.

        Args:
            points: array of shape (n, 3).
            color: uniform RGB color in [0, 1]. Ignored if ``colors`` is provided.
            colors: per-point RGB array of shape (n, 3).
            name: optional tag for later removal with :meth:`remove`.

        Returns:
            self
        """
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if colors is not None and len(colors):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif color is not None:
            pcd.paint_uniform_color(color)
        self._geometries.append((name, pcd))
        return self

    def add_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        *,
        name: str | None = None,
    ) -> Visualizer:
        """Add a pre-built Open3D PointCloud.

        Args:
            pcd: Open3D point cloud object.
            name: optional tag for later removal with :meth:`remove`.

        Returns:
            self
        """
        self._geometries.append((name, pcd))
        return self

    def add_markers(
        self,
        points: np.ndarray,
        color: list[float] | None = None,
        radius: float = 0.1,
        *,
        name: str | None = None,
    ) -> Visualizer:
        """Add individual points rendered as small spheres.

        Args:
            points: array of shape (n, 3).
            color: uniform RGB color in [0, 1]. Defaults to orange.
            radius: sphere radius in metres. Defaults to 0.1.
            name: optional tag for later removal with :meth:`remove`.

        Returns:
            self
        """
        if color is None:
            color = [1, 0.706, 0]
        for pt in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(pt)
            sphere.paint_uniform_color(color)
            self._geometries.append((name, sphere))
        return self

    def add_coordinate_frame(
        self,
        size: float = 1.0,
        origin: list[float] | None = None,
        *,
        name: str | None = None,
    ) -> Visualizer:
        """Add an RGB coordinate frame axes indicator.

        Args:
            size: axis length in metres. Defaults to 1.0.
            origin: position of the frame origin. Defaults to [0, 0, 0].
            name: optional tag for later removal with :meth:`remove`.

        Returns:
            self
        """
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=origin
        )
        self._geometries.append((name, frame))
        return self

    # ── management ────────────────────────────────────────────────────────────

    def remove(self, name: str) -> Visualizer:
        """Remove all geometries added with the given name.

        Args:
            name: tag passed to an earlier ``add_*`` call.

        Returns:
            self
        """
        self._geometries = [(n, g) for n, g in self._geometries if n != name]
        return self

    def clear(self) -> Visualizer:
        """Remove all geometries.

        Returns:
            self
        """
        self._geometries = []
        return self

    # ── display ───────────────────────────────────────────────────────────────

    def show(
        self,
        window_name: str = "pystruct3d",
        width: int = 1280,
        height: int = 720,
        trajectory: str | None = None,
    ) -> None:
        """Open an interactive Open3D viewer with all added geometries.

        Args:
            window_name: title of the viewer window.
            width: window width in pixels. Defaults to 1280.
            height: window height in pixels. Defaults to 720.
            trajectory: path to an Open3D view-trajectory JSON file.
                When provided, opens the animated camera viewer instead of
                the standard one.
        """
        geoms = [g for _, g in self._geometries]
        if not geoms:
            return
        if trajectory is not None:
            o3d.visualization.draw_geometries_with_custom_animation(  # type: ignore
                geoms,
                window_name=window_name,
                width=width,
                height=height,
                optional_view_trajectory_json_file=trajectory,
            )
        else:
            o3d.visualization.draw_geometries(  # type: ignore
                geoms, window_name=window_name, width=width, height=height
            )
