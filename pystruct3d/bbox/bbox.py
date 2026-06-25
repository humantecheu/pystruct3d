# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
from __future__ import annotations

import warnings

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError


class BBox:
    """Class to represent bounding boxes by 8 corner points."""

    def __init__(self, corner_points: np.ndarray | None = None) -> None:  # shape (8, 3)
        if corner_points is None:
            corner_points = np.zeros((8, 3))
        self.corner_points = corner_points
        if np.any(self.corner_points):
            self.order_points()

    def __str__(self) -> str:
        return f"Bounding Box with points {self.corner_points}"

    def order_points(self) -> None:
        """Order corner points counter-clockwise with the length edge at [0]→[1].

        Sorts all 8 corners into lower (z-min) and upper (z-max) groups, orders
        each group counter-clockwise in XY, then rolls the result so that edge
        [0]→[1] runs along the longer horizontal dimension (length).
        """
        # Split points into two groups based on the z-coordinate
        self.corner_points = self.corner_points[self.corner_points[:, 2].argsort()]
        lower_points = self.corner_points[:4]
        upper_points = self.corner_points[4:]

        # Calculate the centroid of each group of points
        lower_centroid = np.mean(lower_points, axis=0)
        upper_centroid = np.mean(upper_points, axis=0)

        # Order each group of points in a counter-clockwise direction
        lower_points = lower_points[
            np.argsort(
                np.arctan2(
                    lower_points[:, 1] - lower_centroid[1],
                    lower_points[:, 0] - lower_centroid[0],
                )
            )
        ]
        upper_points = upper_points[
            np.argsort(
                np.arctan2(
                    upper_points[:, 1] - upper_centroid[1],
                    upper_points[:, 0] - upper_centroid[0],
                )
            )
        ]

        # KNOWN LIMITATION: after CCW ordering the longest edge is not
        # guaranteed to be [0]→[1]; the roll below is a best-effort correction.
        len_edge01 = np.linalg.norm(lower_points[1] - lower_points[0])
        len_edge12 = np.linalg.norm(lower_points[2] - lower_points[1])

        if len_edge01 < len_edge12:
            # change order += 1 i.e. shift point at index 0 to index 1 at lower and upper
            lower_points = np.roll(lower_points, shift=1, axis=0)
            upper_points = np.roll(upper_points, shift=1, axis=0)
            ordered_points = np.vstack((lower_points, upper_points))
            self.corner_points = ordered_points
        else:
            ordered_points = np.vstack((lower_points, upper_points))
            self.corner_points = ordered_points
        # enforce min x / y point to be at [0]
        if (
            lower_points[0, 0] > lower_points[1, 0]
            or lower_points[0, 1] > lower_points[1, 1]
        ):
            lower_points = np.roll(lower_points, shift=2, axis=0)
            upper_points = np.roll(upper_points, shift=2, axis=0)
            ordered_points = np.vstack((lower_points, upper_points))
            self.corner_points = ordered_points
        else:
            ordered_points = np.vstack((lower_points, upper_points))
            self.corner_points = ordered_points
        if self.angle() == 0.0 and lower_points[0, 0] > lower_points[1, 0]:
            lower_points = np.roll(lower_points, shift=2, axis=0)
            upper_points = np.roll(upper_points, shift=2, axis=0)
            ordered_points = np.vstack((lower_points, upper_points))
            self.corner_points = ordered_points

    def _dimensions(self) -> tuple[float, float]:
        """Return (length, width) in a single lower_edges() call."""
        edge_lengths = np.linalg.norm(self.lower_edges(), axis=1)
        return float(np.max(edge_lengths)), float(np.min(edge_lengths))

    # ── Geometric properties ──────────────────────────────────────────────────

    def lower_edges(self) -> np.ndarray:
        """Return the four edge vectors of the lower (z-min) face.

        Returns:
            shape (4, 3) array of edge vectors, ordered to match the
            counter-clockwise corner ordering of the lower face.
        """
        lower_points = self.corner_points[:4]
        lower_points_rolled = np.roll(lower_points, 1, axis=0)
        return lower_points_rolled - lower_points

    def length(self) -> float:
        """Return the length (longer horizontal dimension) of the bounding box.

        ::

              x---------------x
             /|              /|
            x---------------x |
            | |             | |
            | x-------------|-x
            |/              |/
            x---------------x
              <--Length-->

        Returns:
            Length in the same units as the corner points.
        """
        return self._dimensions()[0]

    def width(self) -> float:
        """Return the width (shorter horizontal dimension) of the bounding box.

        ::

                 x-------------------x
                /                   /|
               /                   / |
              /                   /  |
             x-------------------x   |
             |   |               |   |
             |   x---------------|---x   w
             |  /                |  /   i
             | /                 | /   d
             |/                  |/   t
             x-------------------x   h

        Returns:
            Width in the same units as the corner points.
        """
        return self._dimensions()[1]

    def height(self) -> float:
        """Return the height of the bounding box.

        Returns:
            Height in the same units as the corner points.
        """
        zs = self.corner_points[:, 2]
        return float(np.max(zs) - np.min(zs))

    def volume(self) -> float:
        """Return the volume of the bounding box."""
        length, width = self._dimensions()
        return length * width * self.height()

    def angle(self) -> float:
        """Returns the counter-clockwise angle of the bounding box to the x-axis.

        Returns:
            float: angle in degrees
        """
        edges = self.lower_edges()
        lengths = np.linalg.norm(edges, axis=1)
        longest_idx = np.argmax(lengths)

        rad_angle = np.arctan2(edges[longest_idx, 1], edges[longest_idx, 0])
        angle = np.rad2deg(rad_angle)
        # force angle between 0 and 180
        return (angle + 180) % 180

    def dir_vector_norm(self) -> np.ndarray:
        """Return the unit vector along the length edge ([0]→[1]).

        Returns:
            shape (3,) unit vector.
        """
        direction_vec = self.corner_points[1] - self.corner_points[0]
        return direction_vec / np.linalg.norm(direction_vec)

    def get_endpts(self) -> np.ndarray:
        """Return the two midpoints of the length-direction ends of the box.

        Returns:
            np.ndarray: endpoints, shape (2, 3)
        """
        edges = self.lower_edges()
        lengths = np.linalg.norm(edges, axis=1)
        min_idx = np.argmin(lengths)

        lower_points = self.corner_points[:4]
        if min_idx % 2 == 0:
            pt_1 = (lower_points[1, :] + lower_points[2, :]) / 2
            pt_2 = (lower_points[3, :] + lower_points[0, :]) / 2
        else:
            pt_1 = (lower_points[0, :] + lower_points[1, :]) / 2
            pt_2 = (lower_points[2, :] + lower_points[3, :]) / 2

        return np.vstack((pt_1, pt_2))

    def get_center_plane(self) -> np.ndarray:
        """Return the plane equation of the centre plane along the width axis.

        The centre plane bisects the bounding box perpendicular to the width
        direction (i.e., it runs through the middle of the box along its length
        and height axes).

        Returns:
            shape (4,) array ``[a, b, c, d]`` representing ``ax + by + cz + d = 0``,
            where ``[a, b, c]`` is the unit normal in the width direction.
        """
        width_vector = self.corner_points[3] - self.corner_points[0]
        plane_normal = width_vector / np.linalg.norm(width_vector)

        plane_point = self.corner_points[0] + 0.5 * width_vector

        d = np.negative(np.sum(plane_normal * plane_point))
        return np.append(plane_normal, d)

    def get_side_planes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane equations of the two side faces along the width axis.

        Both planes share the same unit normal (the width direction); they
        differ only in their offset term ``d``.

        Returns:
            plane1: shape (4,) ``[a, b, c, d]`` for the face at corner[0].
            plane2: shape (4,) ``[a, b, c, d]`` for the face at corner[3].
        """
        width_vector = self.corner_points[3] - self.corner_points[0]
        plane_normal = width_vector / np.linalg.norm(width_vector)

        d1 = np.negative(np.sum(plane_normal * self.corner_points[0]))
        d2 = np.negative(np.sum(plane_normal * self.corner_points[3]))

        plane_equation1 = np.append(plane_normal, d1)
        plane_equation2 = np.append(plane_normal, d2)

        return plane_equation1, plane_equation2

    # ── Containment ───────────────────────────────────────────────────────────

    def points_in_bbox(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the points inside the bounding box.

        Uses Open3D OrientedBoundingBox for containment testing.

        Args:
            points (np.ndarray): points, numpy array of shape (n, 3)

        Returns:
            np.ndarray: inlier points
            np.ndarray: indices of inlier points
        """
        try:
            obb = self.to_o3d()
            indices = np.asarray(
                obb.get_point_indices_within_bounding_box(
                    o3d.utility.Vector3dVector(points)
                ),
                dtype=np.intp,
            )
            return points[indices], indices
        except (QhullError, ValueError):
            return np.empty((0,)), np.empty((0,))

    def points_in_bbox_2d(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find points within the 2D horizontal footprint of the bounding box.

        Only the four vertical side planes are tested; top and bottom faces
        are ignored. Useful for 2D polygon containment checks in 3D space.

        Implemented as two slab tests on the pair of outward normals n0 / n1
        (one per pair of parallel side faces), avoiding the full 6-plane
        calculation.

        Args:
            points: array of shape (n, 3)

        Returns:
            inlier_points: shape (m, 3)
            indices: integer indices of inlier points into the input array
        """
        cp = self.corner_points
        n0 = np.cross(cp[1] - cp[0], cp[4] - cp[0])
        n0 /= np.linalg.norm(n0)
        n1 = np.cross(cp[2] - cp[1], cp[5] - cp[1])
        n1 /= np.linalg.norm(n1)

        p0 = points @ n0
        p1 = points @ n1

        d0a, d0b = float(cp[0] @ n0), float(cp[2] @ n0)
        d1a, d1b = float(cp[1] @ n1), float(cp[3] @ n1)

        inside = (
            (p0 >= min(d0a, d0b))
            & (p0 <= max(d0a, d0b))
            & (p1 >= min(d1a, d1b))
            & (p1 <= max(d1a, d1b))
        )
        indices = np.flatnonzero(inside)
        return points[indices], indices

    def points_in_bbox_soft(
        self, points: np.ndarray, threshold: float, *, sigma: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Soft Gaussian containment using distance to the nearest bbox surface.

        Interior points always receive pdf = 1.0. Exterior points receive a
        Gaussian probability that decays with distance to the nearest surface
        feature (face, edge, or corner). Points where pdf > ``threshold`` are
        returned as inliers.

        Args:
            points: array of shape (n, 3)
            threshold: probability threshold in (0, 1]
            sigma: Gaussian standard deviation in metres. Controls how quickly
                the probability falls off outside the box. Defaults to 0.1 m
                (≈ 1 σ at 10 cm), which suits typical LiDAR/photogrammetry
                noise levels in scan-to-BIM workflows.

        Returns:
            inlier_points: points where pdf > threshold, shape (m, 3)
            indices: integer indices of inlier points into the input array
            pdf: Gaussian probability for all input points, shape (n,)
        """
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            self.to_o3d(), scale=[1, 1, 1], create_uv_map=False
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        pts_t = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        sdf = scene.compute_signed_distance(pts_t).numpy()
        distance = np.maximum(sdf, 0.0)  # interior points → 0 → pdf = 1.0
        pdf = np.exp(-0.5 * (distance / sigma) ** 2)
        indices = np.flatnonzero(pdf > threshold)
        return points[indices], indices, pdf

    # ── Transforms ────────────────────────────────────────────────────────────

    def rotate(self, angle: float) -> None:
        """Rotate counter-clockwise around the Z-axis through the box centre.

        Args:
            angle (float): angle in degrees
        """
        center = np.mean(self.corner_points, axis=0)
        self.corner_points -= center
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        xy = self.corner_points[:, :2].copy()
        self.corner_points[:, 0] = c * xy[:, 0] - s * xy[:, 1]
        self.corner_points[:, 1] = s * xy[:, 0] + c * xy[:, 1]
        self.corner_points += center
        self.order_points()

    def translate(self, translation_vector: np.ndarray) -> None:  # shape (3, )
        """Translate the bounding box along a given vector.

        Args:
            translation_vector: shape (3,) XYZ translation.
        """
        self.corner_points += translation_vector

    def translate_z(
        self, *, min_z: float | None = None, max_z: float | None = None
    ) -> None:
        """Translate the lower or upper face of the bounding box to a given Z level.

        Args:
            min_z: target Z coordinate for the four lower corners. Defaults to
                the current lower-face Z (no change).
            max_z: target Z coordinate for the four upper corners. Defaults to
                the current upper-face Z (no change).
        """
        if min_z is None:
            min_z = float(self.corner_points[0, 2])
        if max_z is None:
            max_z = float(self.corner_points[4, 2])

        min_zs = np.full((4,), min_z)
        max_zs = np.full((4,), max_z)

        self.corner_points[:4, 2] = min_zs
        self.corner_points[4:8, 2] = max_zs

    def transform_xy(self, xy_dimension: float) -> None:
        """Resize the bounding box to a square footprint of given side length.

        Replaces all 8 corner points so the box has a square XY footprint of
        ``xy_dimension`` × ``xy_dimension``, centred on the current lower and
        upper face centroids.  The Z extent is preserved.

        Args:
            xy_dimension: side length of the square footprint in metres.
        """
        lower_center = np.mean(self.corner_points[:4], axis=0)
        upper_center = np.mean(self.corner_points[4:8], axis=0)
        add_dim = xy_dimension / 2
        self.corner_points[0] = lower_center + np.array([[-add_dim, -add_dim, 0]])
        self.corner_points[1] = lower_center + np.array([[+add_dim, -add_dim, 0]])
        self.corner_points[2] = lower_center + np.array([[+add_dim, +add_dim, 0]])
        self.corner_points[3] = lower_center + np.array([[-add_dim, +add_dim, 0]])
        self.corner_points[4] = upper_center + np.array([[-add_dim, -add_dim, 0]])
        self.corner_points[5] = upper_center + np.array([[+add_dim, -add_dim, 0]])
        self.corner_points[6] = upper_center + np.array([[+add_dim, +add_dim, 0]])
        self.corner_points[7] = upper_center + np.array([[-add_dim, +add_dim, 0]])

    def expand(self, offset: float) -> None:
        """Expand the bounding box outward by ``offset`` in all three dimensions.

        Args:
            offset: distance in metres to expand in each direction (positive
                expands, negative contracts).
        """
        length_vector = self.corner_points[1] - self.corner_points[0]
        length_vector_norm = length_vector / np.linalg.norm(length_vector)
        width_vector = self.corner_points[2] - self.corner_points[1]
        width_vector_norm = width_vector / np.linalg.norm(width_vector)

        length_offset = offset * length_vector_norm
        width_offset = offset * width_vector_norm
        height_offset = offset * np.asarray([0.0, 0.0, 1.0])

        # apply vectors to points
        self.corner_points[0] = (
            self.corner_points[0] - length_offset - width_offset - height_offset
        )
        self.corner_points[1] = (
            self.corner_points[1] + length_offset - width_offset - height_offset
        )
        self.corner_points[2] = (
            self.corner_points[2] + length_offset + width_offset - height_offset
        )
        self.corner_points[3] = (
            self.corner_points[3] - length_offset + width_offset - height_offset
        )
        self.corner_points[4] = (
            self.corner_points[4] - length_offset - width_offset + height_offset
        )
        self.corner_points[5] = (
            self.corner_points[5] + length_offset - width_offset + height_offset
        )
        self.corner_points[6] = (
            self.corner_points[6] + length_offset + width_offset + height_offset
        )
        self.corner_points[7] = (
            self.corner_points[7] - length_offset + width_offset + height_offset
        )

    def axis_align(self) -> None:
        """Axis-align the bounding box by snapping to the nearest 90° boundary."""
        # KNOWN LIMITATION: snaps to nearest 45° boundary, not always the x-axis
        ang = self.angle()
        if ang <= 45:
            self.rotate(-ang)
        elif ang <= 90:
            self.rotate(90 - ang)
        elif ang <= 135:
            self.rotate(-ang + 90)
        else:
            self.rotate(180 - ang)

    def project_into_parent(self, parent_bbox: BBox) -> None:
        """Project bounding box into parent bounding box e.g., door bounding box
        into parent wall. Projection is based on the parent normal, assuming
        that child objects can only be projected along the width i.e., along the main
        surfaces of the parent bounding box.
        Results in the child of same width as parent, surfaces flush to the parents'
        ones. Other dimensions of child might differ slightly if the child is
        rotated onto the parent due to projection.

        Args:
            parent_bbox (bbox): bounding box object of parent
        """
        # rotate bounding box first same angle as parent
        self.rotate(np.negative(self.angle() - parent_bbox.angle()))
        # translate to closest surface
        # Get plane equation ax + by + cz + d = 0
        # get normal vector from parent and normalize
        # no need to calculate normal
        wv = parent_bbox.corner_points[3] - parent_bbox.corner_points[0]
        nv = wv / np.linalg.norm(wv)
        # calculate d: sum of (normal vector x point on plane)
        d1 = np.sum(nv * parent_bbox.corner_points[0])
        # stack normals and d1 to matrices to apply to all 8 corner points
        nv_matr = np.tile(nv, (8, 1))
        d1_matr = np.tile(-d1, (8, 1))
        # distance of point(r, s, u) to plane = |ar + bs + cu + d|
        dist1 = np.sum(np.hstack((self.corner_points * nv_matr, d1_matr)), axis=1)
        # transform points to surface and surface + parent width respectively
        translation = nv_matr * np.negative(np.tile(dist1, (3, 1)).T)
        self.corner_points += translation
        self.corner_points[[2, 3, 6, 7], :] += nv * parent_bbox.width()
        self.order_points()

    def split(self, offset: float = 0.2) -> tuple[BBox, BBox]:
        """Split into two halves along the width axis.

        Args:
            offset: gap between the two halves in metres. Defaults to 0.2.

        Returns:
            Tuple of (first_half, second_half) BBox objects. Self is not modified.
        """
        edges = self.lower_edges()
        half_edges = edges / 2
        transform = half_edges[1] + offset * half_edges[1] / np.linalg.norm(
            half_edges[1]
        )
        first_pts = np.copy(self.corner_points)
        first_pts[[1, 2, 5, 6]] += transform
        second_pts = np.copy(self.corner_points)
        second_pts[[0, 3, 4, 7]] -= transform
        return BBox(first_pts), BBox(second_pts)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_axis_aligned(self, points: np.ndarray) -> None:
        """Fit an axis-aligned bounding box to a set of points.

        Args:
            points (np.ndarray): points, shape (n, 3)
        """
        x_s = points[:, 0]
        y_s = points[:, 1]
        z_s = points[:, 2]

        # fmt:off
        self.corner_points = np.array([
            [np.min(x_s), np.min(y_s), np.min(z_s)],
            [np.max(x_s), np.min(y_s), np.min(z_s)],
            [np.max(x_s), np.max(y_s), np.min(z_s)],
            [np.min(x_s), np.max(y_s), np.min(z_s)],
            [np.min(x_s), np.min(y_s), np.max(z_s)],
            [np.max(x_s), np.min(y_s), np.max(z_s)],
            [np.max(x_s), np.max(y_s), np.max(z_s)],
            [np.min(x_s), np.max(y_s), np.max(z_s)]
        ])
        # fmt:on

    def fit_horizontal_aligned(self, points: np.ndarray) -> None:
        """Fit a minimum-volume Z-aligned bounding box (rotating calipers).

        Projects to XY, builds a convex hull, then tests one candidate rotation
        per hull edge. The AABB min/max for each rotation is computed using only
        the hull vertices (O(h) ≪ O(n)), since they contain all extreme points.
        Z extent is rotation-independent and computed once.

        Args:
            points: shape (n, 3)
        """
        try:
            hull = ConvexHull(points[:, :2])
        except ValueError:
            warnings.warn("No points to fit bounding box.", stacklevel=2)
            return
        except QhullError:
            warnings.warn("Degenerate point set (1-D convex hull).", stacklevel=2)
            return

        # Candidate angles: one per hull edge
        edges_xy = np.diff(points[:, :2][hull.simplices], axis=1).reshape(-1, 2)
        angles = np.abs(np.arctan2(edges_xy[:, 1], edges_xy[:, 0]))  # (E,)

        # Rotate only the unique hull vertices — O(h) not O(n)
        hx, hy = points[:, :2][hull.vertices].T  # (h,) each
        c, s = np.cos(angles), np.sin(angles)  # (E,)

        # Rotated XY coords for all edges × all hull vertices: (E, h)
        xr = c[:, None] * hx[None, :] - s[:, None] * hy[None, :]
        yr = s[:, None] * hx[None, :] + c[:, None] * hy[None, :]

        # AABB extents per candidate rotation
        x_lo, x_hi = xr.min(axis=1), xr.max(axis=1)  # (E,)
        y_lo, y_hi = yr.min(axis=1), yr.max(axis=1)
        z_lo, z_hi = points[:, 2].min(), points[:, 2].max()

        volumes = (x_hi - x_lo) * (y_hi - y_lo) * (z_hi - z_lo)
        n = np.argmin(volumes)
        cn, sn = c[n], s[n]

        # 8 corners of the winning AABB in the rotated frame
        corners_rot = np.array([
            [x_lo[n], y_lo[n], z_lo],
            [x_hi[n], y_lo[n], z_lo],
            [x_hi[n], y_hi[n], z_lo],
            [x_lo[n], y_hi[n], z_lo],
            [x_lo[n], y_lo[n], z_hi],
            [x_hi[n], y_lo[n], z_hi],
            [x_hi[n], y_hi[n], z_hi],
            [x_lo[n], y_hi[n], z_hi],
        ])

        # Inverse rotation (R^{-1} = R^T for z-axis rotation):
        # x_orig = x_rot * cos + y_rot * sin  # noqa: ERA001
        # y_orig = -x_rot * sin + y_rot * cos  # noqa: ERA001
        corners = corners_rot.copy()
        corners[:, 0] = corners_rot[:, 0] * cn + corners_rot[:, 1] * sn
        corners[:, 1] = -corners_rot[:, 0] * sn + corners_rot[:, 1] * cn

        self.corner_points = corners
        self.order_points()

    # TODO: implement fit_minimal — minimum bounding box (no Z-axis constraint)
    # def fit_minimal(self) -> None:
    #     raise NotImplementedError

    # ── Serialisation & I/O ───────────────────────────────────────────────────

    def as_np_array(self) -> np.ndarray:
        """Return the corner points as a numpy array.

        Returns:
            shape (8, 3) array of corner points (same object as ``self.corner_points``).
        """
        return self.corner_points

    def to_o3d(self) -> o3d.geometry.OrientedBoundingBox:
        """Convert to an Open3D OrientedBoundingBox.

        The three OBB axes are derived from edges [0]→[1] (length),
        [0]→[3] (width), and [0]→[4] (height).
        """
        center = np.mean(self.corner_points, axis=0)
        l_vec = self.corner_points[1] - self.corner_points[0]
        w_vec = self.corner_points[3] - self.corner_points[0]
        h_vec = self.corner_points[4] - self.corner_points[0]
        length = np.linalg.norm(l_vec)
        width = np.linalg.norm(w_vec)
        height = np.linalg.norm(h_vec)
        R = np.column_stack([l_vec / length, w_vec / width, h_vec / height])
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = center
        obb.R = R
        obb.extent = np.array([length, width, height])
        return obb

    def to_dict(self) -> dict:
        """Serialise corner points to a JSON-compatible dict."""
        return {"corner_points": self.corner_points.tolist()}

    def save(self, path: str) -> None:
        """Save corner points to a ``.npy`` file."""
        np.save(path, self.corner_points)

    # ── Classmethods (factories) ──────────────────────────────────────────────

    @classmethod
    def from_params(
        cls,
        position: np.ndarray,
        lwh: tuple[float, float, float] | np.ndarray,
        yaw: float = 0.0,
        *,
        origin: str = "center",
    ) -> BBox:
        """Construct from parametric position, dimensions, and yaw.

        Args:
            position: (3,) xyz coordinates. Interpreted as the box centre when
                ``origin="center"`` (default), or as the minimum corner
                (bottom-left-front in the pre-rotation frame) when
                ``origin="corner"``.
            lwh: (length, width, height) — length is the longer horizontal dim.
            yaw: rotation around the Z-axis in radians (CCW). Defaults to 0.0.
            origin: ``"center"`` or ``"corner"``. Defaults to ``"center"``.

        Returns:
            BBox with the described geometry.

        Raises:
            ValueError: if ``origin`` is not ``"center"`` or ``"corner"``.
        """
        length, w, h = lwh
        hl, hw, hh = length / 2, w / 2, h / 2

        if origin == "center":
            center = np.asarray(position, dtype=float)
        elif origin == "corner":
            center = np.asarray(position, dtype=float) + np.array([hl, hw, hh])
        else:
            raise ValueError(f"origin must be 'center' or 'corner', got {origin!r}")

        corners = np.array(
            [
                [-hl, -hw, -hh],
                [hl, -hw, -hh],
                [hl, hw, -hh],
                [-hl, hw, -hh],
                [-hl, -hw, hh],
                [hl, -hw, hh],
                [hl, hw, hh],
                [-hl, hw, hh],
            ],
            dtype=float,
        )
        c, s = np.cos(yaw), np.sin(yaw)
        xy = corners[:, :2].copy()
        corners[:, 0] = c * xy[:, 0] - s * xy[:, 1]
        corners[:, 1] = s * xy[:, 0] + c * xy[:, 1]
        corners += center
        return cls(corners)

    @classmethod
    def from_o3d(cls, obb: o3d.geometry.OrientedBoundingBox) -> BBox:
        """Construct a BBox from an Open3D OrientedBoundingBox."""
        corners = np.asarray(obb.get_box_points())
        return cls(corners)

    @classmethod
    def from_dict(cls, d: dict) -> BBox:
        """Reconstruct from a dict produced by ``to_dict()``."""
        return cls(np.asarray(d["corner_points"]))

    @classmethod
    def load(cls, path: str) -> BBox:
        """Load corner points from a ``.npy`` file written by ``save()``."""
        return cls(np.load(path))

    # ── Deprecated ────────────────────────────────────────────────────────────

    def points_in_bbox_probability(
        self,
        points: np.ndarray,
        probability_threshold: float = 0,
        *,
        in_2d: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deprecated since 0.4.1. Use points_in_bbox(), points_in_bbox_2d(), or points_in_bbox_soft().

        .. deprecated:: 0.4.1

        - Hard 3D containment  → points_in_bbox(points)
        - 2D footprint test    → points_in_bbox_2d(points)
        - Soft Gaussian        → points_in_bbox_soft(points, threshold)
        """
        warnings.warn(
            "points_in_bbox_probability is deprecated. "
            "Use points_in_bbox() for hard 3D containment, "
            "points_in_bbox_2d() for 2D footprint testing, or "
            "points_in_bbox_soft(points, threshold) for Gaussian soft membership.",
            DeprecationWarning,
            stacklevel=2,
        )
        if in_2d:
            return self.points_in_bbox_2d(points)
        if probability_threshold > 0:
            return self.points_in_bbox_soft(points, probability_threshold)
        return self.points_in_bbox(points)

    def from_cv4aec(self, cv4aec_dict: dict) -> None:
        """Create bounding boxes from cv4aec parameters.

        .. deprecated:: 0.6.0
            Use :func:`pystruct3d.io.cv4aec.bbox_from_cv4aec` instead.

        Args:
            cv4aec_dict (dict): dictionary of wall, door or column parameter
        """
        warnings.warn(
            "BBox.from_cv4aec() is deprecated; use pystruct3d.io.cv4aec.bbox_from_cv4aec() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from pystruct3d.io.cv4aec import bbox_from_cv4aec

        box = bbox_from_cv4aec(cv4aec_dict)
        self.corner_points = box.corner_points

    def to_cv4aec(
        self, output_style: str = "start_pt", element_id: str = "0", host_id: str = "0"
    ) -> dict:
        """Returns the bounding box geometry as a dictionary of cv4aec style parameters.

        .. deprecated:: 0.6.0
            Use :func:`pystruct3d.io.cv4aec.bbox_to_cv4aec` instead.

        Args:
            output_style (str, optional): Either start_pt for walls or loc for doors and columns. Defaults to "start_pt".
            element_id (str, optional): ID of element, IFC ID can be used. Defaults to "0".
            host_id (str, optional): ID of host element. Defaults to "0".

        Returns:
            dict: Dictionary according to output style specified
        """
        warnings.warn(
            "BBox.to_cv4aec() is deprecated; use pystruct3d.io.cv4aec.bbox_to_cv4aec() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from pystruct3d.io.cv4aec import bbox_to_cv4aec

        return bbox_to_cv4aec(self, output_style, element_id, host_id)
