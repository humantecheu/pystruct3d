import numpy as np
from scipy.spatial import ConvexHull


class BBox:
    """Class to represent bounding boxes by 8 corner points. Note that other classes exist to fit the bounding boxes to the data with specific methods."""

    def __init__(self, corner_points=np.zeros((8, 3))) -> None:  # shape (8, 3)
        self.corner_points = corner_points
        # ensure the corners are ordered as the BBox is initialized
        if np.any(self.corner_points):
            self.order_points()

    def __str__(self):
        string = f"Bounding Box with points {self.corner_points}"
        return string

    def order_points(self):
        """
        Orders the points of a 3D bounding box aligned with the z-axis in a counter-clockwise direction
        starting from the left-most point.
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

        # Combine the ordered points
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
        rot_mat = np.array(
            [
                [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z**2 + c],
            ]
        )
        # translate points to the origin
        self.corner_points -= center
        # apply the rotation
        self.corner_points = np.dot(self.corner_points, rot_mat.T)
        # Translate the points back to the original position
        self.corner_points += center

    def points_in_bbox_probability(self, points: np.ndarray):
        def calculate_plane_normals():
            n1 = np.cross(
                self.corner_points[1] - self.corner_points[0],
                self.corner_points[4] - self.corner_points[0],
            )
            n1 = n1 / np.linalg.norm(n1)
            n2 = np.cross(
                self.corner_points[2] - self.corner_points[1],
                self.corner_points[5] - self.corner_points[1],
            )
            n2 = n2 / np.linalg.norm(n2)
            n3 = np.cross(
                self.corner_points[3] - self.corner_points[2],
                self.corner_points[6] - self.corner_points[2],
            )
            n3 = n3 / np.linalg.norm(n3)
            n4 = np.cross(
                self.corner_points[0] - self.corner_points[3],
                self.corner_points[7] - self.corner_points[3],
            )
            n4 = n4 / np.linalg.norm(n4)
            n5 = np.cross(
                self.corner_points[3] - self.corner_points[0],
                self.corner_points[1] - self.corner_points[0],
            )
            n5 = n5 / np.linalg.norm(n5)
            n6 = np.cross(
                self.corner_points[5] - self.corner_points[4],
                self.corner_points[7] - self.corner_points[4],
            )
            n6 = n6 / np.linalg.norm(n6)

            return np.array([n1, n2, n3, n4, n5, n6])

        normals = calculate_plane_normals()

        def calculate_relative_position():
            p1 = np.dot(points - self.corner_points[0], normals[0])
            p2 = np.dot(points - self.corner_points[1], normals[1])
            p3 = np.dot(points - self.corner_points[2], normals[2])
            p4 = np.dot(points - self.corner_points[3], normals[3])
            p5 = np.dot(points - self.corner_points[0], normals[4])
            p6 = np.dot(points - self.corner_points[4], normals[5])

            return np.where(np.vstack((p1, p2, p3, p4, p5, p6)) > 0, 1, 0).T

        positions = calculate_relative_position()
        print(positions)

        return np.where(
            # (positions[:, 0] == 0)
            # & (positions[:, 1] == 0)
            # & (positions[:, 2] == 0)
            # & (positions[:, 3] == 0)
            # & (positions[:, 4] == 0)
            # & (positions[:, 5] == 0)
        )[0]

    def points_in_BBox(self, points: np.ndarray, tolerance=1e-12):
        """find the points inside a bounding box

        Args:
            points (np.ndarray): points, numpy array of shape (n, 3)
            tolerance (float, optional): tolerance of points distance to bounding box. Defaults to 1e-12.

        Returns:
            np.ndarray: points inliers
            np.ndarray: indices of inlier points
        """

        try:
            # convex hull
            hull = ConvexHull(self.corner_points)

            # Get array of boolean values indicating in hull if True
            in_hull = np.all(
                np.add(np.dot(points, hull.equations[:, :-1].T), hull.equations[:, -1])
                <= tolerance,
                axis=1,
            )  # tolerance could be set to zero, not tested

            # Get the actual points inside the box
            points_in_box = points[in_hull]
            indices = np.where(in_hull == True)

            return points_in_box, indices
        except:
            print("-- trying to construct empty convex hull, passing ...")
            return np.empty((0,)), np.empty((0,))

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

    def angle(self):
        """Returns the counter-clockwise angle of the bounding box to the x-axis.

        Returns:
            float: angle in degrees
        """
        edges = self.lower_edges()
        lengths = np.linalg.norm(edges, axis=1)
        longest_idx = np.argmax(lengths)

        rad_angle = np.arctan2(edges[longest_idx, 1], edges[longest_idx, 0])
        angle = np.rad2deg(rad_angle)
        angle = (angle + 360) % 360

        return angle

    def split_bounding_box(self):
        """Splits the bounding box into two. Modifies self, returns the other box

        Returns:
            BBox: other bounding box
        """
        # take all lower points, calculate edge lengths, return largest as length
        edges = self.lower_edges()
        lengths = np.linalg.norm(edges, axis=1)
        longest_idx = np.argmax(lengths)
        half_edges = edges / 2

        # calculate the transformation matrix
        transform = half_edges[longest_idx]

        transform_mat = np.zeros((8, 3))
        transform_mat[[1, 2, 5, 6]] = transform

        other_transform_mat = np.zeros((8, 3))
        other_transform_mat[::] = transform

        self.corner_points += transform_mat

        # add other box
        other_points = np.copy(self.corner_points)
        other_points -= other_transform_mat
        other_box = BBox(other_points)

        return other_box

    def axis_align(self):
        """Axis aligns the bounding box. Calculates the minimum value against the
        x-axis, then rotates the box with this value."""
        # copy and roll one set of points so that the order of the rolled is n + 1
        points_rolled = np.roll(self.corner_points, 1, axis=0)
        edges = points_rolled - self.corner_points
        # reduce to 2 dimension
        edges_xy = edges[:, :2]
        # calculate angles
        angles = np.abs(np.arctan2(edges_xy[:, 1], edges_xy[:, 0]))
        angle = np.rad2deg(np.min(angles))
        # rotate the minimum angle
        angle = angle % 90
        if angle <= 45:
            self.rotate(-angle)
        else:
            self.rotate(90 - angle)

    def as_np_array(self):
        return self.corner_points

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

    def fit_horizontal_aligned(self, points):
        """Fits a minimum horizontal aligned bounding box to the points. The minimum
        bounding box is identified as the one with the minimal volume. The fitting is
        done based on the 2D projection of the points in the xy plane. A convex hull is
        fitted first, and the points are rotated along the x-axis per edge. Then an
        axis aligned bounding box is fitted to the rotated points.

        Args:
            points (np.ndarray): points, shape (n, 3)
        """
        # delete 0s from Z dimension
        points2d = points[:, :2]
        #
        hull = ConvexHull(points2d)

        conv_hull_points = points2d[hull.simplices]

        # print(f"conv_hull_points{conv_hull_points.shape}")

        edges_xy = np.diff(conv_hull_points, axis=1).reshape(-1, 2)
        print(edges_xy.shape)

        angles = np.abs(np.arctan2(edges_xy[:, 1], edges_xy[:, 0]))

        print(np.rad2deg(angles))
        candidate_volumes = np.array([])
        rotation_matrices = np.empty((0, 3, 3))
        candidate_boxes = np.empty((0, 8, 3))
        for angle in angles:
            # Create a rotation matrix for the specified angle and axis
            axis = np.array([0, 0, 1])
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            axis = axis / np.linalg.norm(axis)
            x, y, z = axis
            # fmt:off
            rot_mat = np.array(
                [
                    [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
                    [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
                    [t * x * z - s * y, t * y * z + s * x, t * z**2 + c],
                ]
            )
            rotation_matrices = np.vstack((rotation_matrices, rot_mat.reshape(1, 3, 3)))
            # fmt:on
            # apply the rotation
            rot_points = np.dot(points, rot_mat.T)

            # fit axis aligned bounding box
            box_canidate = BBox()
            box_canidate.fit_axis_aligned(rot_points)
            box_points = box_canidate.corner_points
            candidate_boxes = np.vstack((candidate_boxes, box_points.reshape(1, 8, 3)))
            cand_volume = box_canidate.volume()
            candidate_volumes = np.append(candidate_volumes, cand_volume)

        print(candidate_volumes)
        print(f"shape of rot matrices, {rotation_matrices.shape}")

        min_idx = np.argmin(candidate_volumes)
        min_box_points = candidate_boxes[min_idx]
        min_box_points = np.dot(
            min_box_points, np.linalg.inv(rotation_matrices[min_idx]).T
        )
        self.corner_points = min_box_points

        return self

    def fit_minimal():
        pass

    def bbox_from_verts(
        self,
        verts: np.ndarray,  # n,
    ) -> np.ndarray:
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

        # remove points between min_z, max_z
        min_z = np.amin(corner_pts[:, 2])
        max_z = np.amax(corner_pts[:, 2])

        rm_indices = np.where((corner_pts[:, 2] > min_z) & (corner_pts[:, 2] < max_z))
        corner_pts = np.delete(corner_pts, rm_indices, axis=0)

        # remove points on edges in between min and max points (could be door points etc.)

        # find centroid
        # centrd = np.median(bbox, axis=0)
        centrd = np.mean(corner_pts, axis=0)

        # find 8 points furthest away from centroid

        diffs = np.subtract(corner_pts, centrd)

        dists = np.linalg.norm(diffs, axis=1)

        # take 8 points with maximum distances to centroid
        corner_pts_idx = (-dists).argsort()[:8]

        corner_pts = corner_pts[corner_pts_idx]

        self.corner_points = corner_pts
        self.order_points()
