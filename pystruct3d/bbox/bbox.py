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

    def __str__(self):
        string = f"Bounding Box with points {self.corner_points}"
        return string

    def order_points(self):
        """
        Orders the points of a 3D bounding box aligned with the z-axis in
        counter-clockwise direction, with condition that edge [0] - [1]
        is along the length == longer horizontal dimension
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

        # BUG
        # check if edge [0] - [1] is longer than [1] - [2]
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
        if self.angle() == 0.0:
            if lower_points[0, 0] > lower_points[1, 0]:
                lower_points = np.roll(lower_points, shift=2, axis=0)
                upper_points = np.roll(upper_points, shift=2, axis=0)
                ordered_points = np.vstack((lower_points, upper_points))
                self.corner_points = ordered_points

    def rotate(self, angle: float):  # angle in degrees
        """Rotates the bounding box counter-clockwise around the z-axis and the bounding box center

        Args:
            angle (float): angle in degrees
        """

        # find the center
        center = np.mean(self.corner_points, axis=0)
        # perform the rotation
        rad_angle = np.deg2rad(angle)
        # Create a rotation matrix for the specified angle and axis
        axis = np.array([0, 0, 1])
        c = np.cos(rad_angle)
        s = np.sin(rad_angle)
        t = 1 - c
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        rot_mat = np.array([
            [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z**2 + c],
        ])
        # translate points to the origin
        self.corner_points -= center
        # apply the rotation
        self.corner_points = np.dot(self.corner_points, rot_mat.T)
        # Translate the points back to the original position
        self.corner_points += center
        self.order_points()

    def points_in_bbox_probability(
        self,
        points: np.ndarray,
        probability_threshold: float = 0,
        in_2d: bool = False,
    ):
        """Deprecated. Use points_in_bbox(), points_in_bbox_2d(), or points_in_bbox_soft().

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

    def points_in_bbox(
        self, points: np.ndarray, tolerance: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the points inside the bounding box.

        Uses Open3D OrientedBoundingBox for containment testing.
        The ``tolerance`` parameter is retained for API compatibility but is
        not forwarded to the Open3D backend.

        Args:
            points (np.ndarray): points, numpy array of shape (n, 3)
            tolerance (float, optional): kept for API compatibility. Defaults to 1e-12.

        Returns:
            np.ndarray: inlier points
            np.ndarray: indices of inlier points
        """
        try:
            obb = self.to_o3d()
            indices = np.asarray(
                obb.get_point_indices_within_bounding_box(
                    o3d.utility.Vector3dVector(points)
                )
            )
            return points[indices], indices
        except (QhullError, ValueError, Exception):
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

    def translate(self, translation_vector: np.ndarray):  # shape (3, )
        """translate the bounding box along a given vector

        Args:
            translation_vector (np.ndarray): translation vector
        """
        self.corner_points += translation_vector

    def translate_z(self, **kwargs):
        """translates the bounding box 4 lower points to min_z or the four top points to max_z respectively

        Kwargs:
            min_z (float): minimum z value
            max_z (float): maximum z value

        """

        # get keyword arguments
        min_z = kwargs.get("min_z", self.corner_points[0, 2])
        max_z = kwargs.get("max_z", self.corner_points[4, 2])

        min_zs = np.full((4,), min_z)
        max_zs = np.full((4,), max_z)

        self.corner_points[:4, 2] = min_zs
        self.corner_points[4:8, 2] = max_zs

    def transform_xy(self, xy_dimension: float):
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

    def expand(self, offset):
        """Expand the bounding box to either direction with a given offset"""
        length_vector = self.corner_points[1] - self.corner_points[0]
        length_vector_norm = length_vector / np.linalg.norm(length_vector)
        widht_vector = self.corner_points[2] - self.corner_points[1]
        width_vector_norm = widht_vector / np.linalg.norm(widht_vector)

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

    def lower_edges(self):
        lower_points = self.corner_points[:4]
        lower_points_rolled = np.roll(lower_points, 1, axis=0)
        edges = lower_points_rolled - lower_points

        return edges

    def length(self):
        """returns the length (base plane), always larger dimension

        Returns:
            float: length
        """
        """      
          x---------------x 
         /|              /|
        x---------------x |
        | |             | |
        | x-------------|-x 
        |/              |/ 
        x---------------x
          <--Length-->     
        """
        edges = self.lower_edges()
        lengths = np.linalg.norm(edges, axis=1)
        length = np.max(lengths)

        return length

    def width(self):
        """returns the width (base plane), always smaller dimension

        Returns:
            float: width
        """
        """     x-------------------x
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
        """
        edges = self.lower_edges()
        lengths = np.linalg.norm(edges, axis=1)
        width = np.min(lengths)

        return width

    def height(self):
        """Returns the height of the bounding box

        Returns:
            float: height
        """

        zs = self.corner_points[:, 2]

        height = np.max(zs) - np.min(zs)

        return height

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
        # if angle != 180:
        #     angle = (angle + 180) % 180
        angle = (angle + 180) % 180
        # angle = (angle + 360) % 360
        return angle

    def dir_vector_norm(self) -> np.ndarray:
        direction_vec = self.corner_points[1] - self.corner_points[0]
        dir_norm = direction_vec / np.linalg.norm(direction_vec)
        return dir_norm

    def project_into_parent(self, parent_bbox):
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

    def split_bounding_box(self, offset=0.2):
        """Splits the bounding box into two. Modifies self, returns the other box

        Returns:
            BBox: other bounding box
        """
        # get lower edges
        edges = self.lower_edges()
        half_edges = edges / 2

        # calculate the transformation matrix
        transform = half_edges[1] + offset * half_edges[1] / np.linalg.norm(
            half_edges[1]
        )

        transform_mat = np.zeros((8, 3))
        transform_mat[[1, 2, 5, 6]] = transform
        # transform self
        self.corner_points += transform_mat

        other_transform_mat = np.zeros((8, 3))
        other_transform_mat[::] = transform

        # add other box and transform
        other_points = np.copy(self.corner_points)
        other_points -= other_transform_mat
        other_box = BBox(other_points)

        return other_box

    def axis_align(self):
        """Axis aligns the bounding box. Calculates the minimum value against the
        x-axis, then rotates the box with this value."""
        # BUG: aligns bounding box to x-axis always
        ang = self.angle()
        if ang <= 45:
            self.rotate(-ang)
        elif ang <= 90:
            self.rotate(90 - ang)
        elif ang <= 135:
            self.rotate(-ang + 90)
        else:
            self.rotate(180 - ang)

    def as_np_array(self):
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

    @classmethod
    def from_o3d(cls, obb: o3d.geometry.OrientedBoundingBox) -> BBox:
        """Construct a BBox from an Open3D OrientedBoundingBox."""
        corners = np.asarray(obb.get_box_points())
        return cls(corners)

    def get_endpts(self):
        """Returns the endpoints

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

        endpts = np.vstack((pt_1, pt_2))

        return endpts

    def get_center_plane(self):
        width_vector = self.corner_points[3] - self.corner_points[0]
        plane_normal = width_vector / np.linalg.norm(width_vector)

        plane_point = self.corner_points[0] + 0.5 * width_vector

        # plane equation: ax + by + cx + d = 0

        d = np.negative(np.sum(plane_normal * plane_point))
        plane_equation = np.append(plane_normal, d)
        return plane_equation

    def get_side_planes(self):
        width_vector = self.corner_points[3] - self.corner_points[0]
        plane_normal = width_vector / np.linalg.norm(width_vector)

        # plane equation: ax + by + cx + d = 0

        d1 = np.negative(np.sum(plane_normal * self.corner_points[0]))
        d2 = np.negative(np.sum(plane_normal * self.corner_points[3]))

        plane_equation1 = np.append(plane_normal, d1)
        plane_equation2 = np.append(plane_normal, d2)

        return plane_equation1, plane_equation2

    def volume(self):
        """calculate the volume of the bounding box

        Returns:
            float: volume
        """
        box_length = self.length()
        box_width = self.width()
        box_heigth = self.height()

        volume = box_length * box_width * box_heigth

        return volume

    def fit_axis_aligned(self, points: np.ndarray):
        """fits an axis aligned bounding box to a set of points

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

    def fit_horizontal_aligned(self, points: np.ndarray) -> "BBox":
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
            return self
        except QhullError:
            warnings.warn("Degenerate point set (1-D convex hull).", stacklevel=2)
            return self

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
        # x_orig = x_rot * cos + y_rot * sin
        # y_orig = -x_rot * sin + y_rot * cos
        corners = corners_rot.copy()
        corners[:, 0] = corners_rot[:, 0] * cn + corners_rot[:, 1] * sn
        corners[:, 1] = -corners_rot[:, 0] * sn + corners_rot[:, 1] * cn

        self.corner_points = corners
        self.order_points()
        return self

    def fit_minimal(self) -> "BBox":
        raise NotImplementedError

    def bbox_from_verts(
        self,
        verts: np.ndarray,  # n,
        force_cuboid: bool = True,
    ) -> BBox:
        """Function to get the bounding box from vertices. Vertices are the
        points of an IFC geometry shape as a np.ndarray of shape n, . The vertices
        can be obtained from the IFC geometry using IfcOpenShell as follows:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.asarray(shape.geometry.verts)
        where element is an IfcElement e.g., IfcWall
        Note, that you have to provide the element using IfcOpenShell e.g.,
        filtered from an IFC file.

        Args:
            verts (np.ndarray): input vertices of shape n,

        Returns:
            bounding box (np.ndarray): bouding box of shape (8, 3)
        """

        # reshape to n, 3
        orig_shape = verts.shape[0]
        verts = verts.reshape((int(orig_shape / 3), 3))
        corner_pts = verts[verts[:, 2].argsort()]
        # vertices from IFC may result in arbitrary non-cuboid bounding boxes
        if force_cuboid:
            # fit a bounding box to enforce cuboid-shaped bounding box
            self.fit_horizontal_aligned(verts)
        else:
            # if non-cuboid shape is acceptable, find extreme and use as corner points
            # min, max and mean of XYZ
            min_x = np.amin(corner_pts[:, 0])
            max_x = np.amax(corner_pts[:, 0])
            mean_x = (min_x + max_x) / 2

            min_y = np.amin(corner_pts[:, 1])
            max_y = np.amax(corner_pts[:, 1])
            mean_y = (min_y + max_y) / 2

            min_z = np.amin(corner_pts[:, 2])
            max_z = np.amax(corner_pts[:, 2])
            mean_z = (min_z + max_z) / 2

            # find centroid
            centrd = np.asarray([mean_x, mean_y, mean_z])

            # find 8 points furthest away from centroid
            diffs = np.subtract(corner_pts, centrd)
            dists = np.linalg.norm(diffs, axis=1)

            # take 8 points with maximum distances to centroid
            corner_pts_idx = (-dists).argsort()[:8]
            corner_pts = corner_pts[corner_pts_idx]

            self.corner_points = corner_pts
            self.order_points()

        return self

    def from_cv4aec(self, cv4aec_dict: dict):
        """Create bounding boxes from cv4aec parameters

        Args:
            cv4aec_dict (dict): dictionary of wall, door or column parameter
        """
        # distinguish cases
        # 1. case: start_pt, end_pt, width, height
        if "start_pt" in cv4aec_dict:
            start_pt = cv4aec_dict["start_pt"]
            end_pt = cv4aec_dict["end_pt"]
            width = cv4aec_dict["width"]
            height = cv4aec_dict["height"]

            start_vec = np.asarray(start_pt)
            end_vec = np.asarray(end_pt)
            center_dir = end_vec - start_vec

            # we know that the base plane of the bounding box is horizontal
            offset_dir = np.cross(center_dir, np.asarray([0, 0, 1]))
            offset_norm = offset_dir / np.linalg.norm(offset_dir)
            self.corner_points[0] = start_vec + 0.5 * width * offset_norm
            self.corner_points[1] = start_vec - 0.5 * width * offset_norm
            self.corner_points[2] = end_vec + 0.5 * width * offset_norm
            self.corner_points[3] = end_vec - 0.5 * width * offset_norm
            self.corner_points[4] = self.corner_points[0]
            self.corner_points[5] = self.corner_points[1]
            self.corner_points[6] = self.corner_points[2]
            self.corner_points[7] = self.corner_points[3]
            self.corner_points[:-4:, 2] += height
            self.order_points()

        elif "loc" in cv4aec_dict:
            location = cv4aec_dict["loc"]
            bx_width = cv4aec_dict["width"]
            bx_depth = cv4aec_dict["depth"]
            bx_height = cv4aec_dict["height"]
            rotation = cv4aec_dict["rotation"]

            loc_vec = np.asarray(location)
            self.corner_points[0] = loc_vec - np.asarray([
                0.5 * bx_width,
                0.5 * bx_depth,
                0.0,
            ])
            self.corner_points[1] = loc_vec + np.asarray([
                0.5 * bx_width,
                -0.5 * bx_depth,
                0,
            ])
            self.corner_points[2] = loc_vec + np.asarray([
                0.5 * bx_width,
                0.5 * bx_depth,
                0,
            ])
            self.corner_points[3] = loc_vec + np.asarray([
                -0.5 * bx_width,
                0.5 * bx_depth,
                0,
            ])
            self.corner_points[4] = self.corner_points[0]
            self.corner_points[5] = self.corner_points[1]
            self.corner_points[6] = self.corner_points[2]
            self.corner_points[7] = self.corner_points[3]
            self.corner_points[:-4:, 2] += bx_height
            self.order_points()
            if round(cv4aec_dict["rotation"], 0) != 0:
                self.rotate(rotation)

        else:
            warnings.warn(
                "from_cv4aec: no valid key found in dict (expected 'start_pt' or 'loc').",
                stacklevel=2,
            )

    def to_cv4aec(self, output_style="start_pt", element_id="0", host_id="0"):
        """Returns the bounding box geometry as a dictionary of cv4aec style parameters.
        Output_styles:
        "start_pt": for walls
        "loc" for doors and columns

        Args:
            output_style (str, optional): Either start_pt for walls or loc for doors and columns. Defaults to "start_pt".
            element_id (str, optional): ID of element, IFC ID can be used. Defaults to "0".
            host_id (str, optional): ID of host element. Defaults to "0".

        Returns:
            dict: Dictionary accourding to output style specified
        """
        if output_style == "loc":
            data_dict = {
                "id": element_id,
                "width": self.length(),
                "depth": self.width(),
                "height": self.height(),
                "loc": list(np.mean(self.corner_points[0:3], axis=0)),
                "rotation": self.angle(),
                "host_id": host_id,
            }
        elif output_style == "start_pt":
            side_vector = self.corner_points[2] - self.corner_points[1]
            to_base_vec = 0.5 * side_vector
            data_dict = {
                "id": element_id,
                "start_pt": list(self.corner_points[0] + to_base_vec),
                "end_pt": list(self.corner_points[1] + to_base_vec),
                "width": self.width(),
                "height": self.height(),
            }

        return data_dict
