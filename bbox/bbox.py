import numpy as np
from scipy.spatial import ConvexHull

class bbox:
    """Class to represent bounding boxes by 8 corner points. Note that other classes exist to fit the bounding boxes to the data with specific methods.
    """

    def __init__(self, 
                 corner_points: np.ndarray # shape (8, 3)
                 ) -> None:
        self.corner_points = corner_points

    def __str__(self):
        string = f"Bounding Box with points {self.corner_points}"
        return string
    
    def order_points(self):
        """
        Orders the points of a 3D bounding box aligned with the z-axis in a counter-clockwise direction 
        starting from the left-most point.
        """
        # Split points into two groups based on the z-coordinate
        self.corner_points = self.corner_points[self.corner_points[:,2].argsort()]
        lower_points = self.corner_points[:4]
        upper_points = self.corner_points[4:]
        
        # Calculate the centroid of each group of points
        lower_centroid = np.mean(lower_points, axis=0)
        upper_centroid = np.mean(upper_points, axis=0)
        
        # Order each group of points in a counter-clockwise direction
        lower_points = lower_points[np.argsort(np.arctan2(lower_points[:,1] - lower_centroid[1], lower_points[:,0] - lower_centroid[0]))]
        upper_points = upper_points[np.argsort(np.arctan2(upper_points[:,1] - upper_centroid[1], upper_points[:,0] - upper_centroid[0]))]
        
        # Combine the ordered points
        ordered_points = np.vstack((lower_points, upper_points))
        self.corner_points = ordered_points
    
    def rotate(self, 
               angle: float # angle in degrees
               ):
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
            [t*x**2+c, t*x*y-s*z, t*x*z+s*y],
            [t*x*y+s*z, t*y**2+c, t*y*z-s*x],
            [t*x*z-s*y, t*y*z+s*x, t*z**2+c]
        ])
        # translate points to the origin
        self.corner_points -= center
        # apply the rotation
        self.corner_points = np.dot(self.corner_points, rot_mat.T)
        # Translate the points back to the original position
        self.corner_points += center
    
    def points_in_bbox(self, 
                       points: np.ndarray,
                       tolerance=1e-12
                       ):
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
            in_hull = np.all(np.add(np.dot(points, hull.equations[:,:-1].T),
                            hull.equations[:,-1]) <= tolerance, axis=1) # tolerance could be set to zero, not tested
            

            # Get the actual points inside the box
            points_in_box = points[in_hull]
            indices = np.where(in_hull == True)

            return points_in_box, indices
        except:
            print("-- trying to construct empty convex hull, passing ...")
            return np.empty((0,)), np.empty((0,))
        
    def translate(self, 
                  translation_vector: np.ndarray # shape (3, )
                  ):
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

        min_zs = np.full((4, ), min_z)
        max_zs = np.full((4, ), max_z)

        self.corner_points[:4, 2] = min_zs
        self.corner_points[4:8, 2] = max_zs

    def width(self):
        """returns the width (base plane), always larger dimension

        Returns:
            float: width
        """
        """      
          x---------------x 
         /|              /|
        x---------------x |
        | |             | |
        | x-------------|-x 
        |/              |/ 
        x---------------x
          <--WIDTH-->     
        """

        # take all lower points, calculate edge lengths, return largest as width        
        lower_points = self.corner_points[: 4]
        lower_points_rolled = np.roll(lower_points, 1, axis=0)
        edges = lower_points_rolled - lower_points
        widths = np.linalg.norm(edges, axis=1)
        width = np.max(widths)

        return width
        

    def split_bounding_box(self):
        """Splits the bounding box into two. Modifies self, returns the other box

        Returns:
            bbox: other bounding box
        """

        # take all lower points, calculate edge lengths, return largest as width        
        points_rolled = np.roll(self.corner_points, 1, axis=0)
        edges = points_rolled - self.corner_points
        widths = np.linalg.norm(edges, axis=1)
        longest_idx = np.argmax(widths)
        half_edges = edges / 2

        # set every 2nd row to zero starting from the first row
        if longest_idx == 0:
            transform = half_edges[0]

            transform_mat = np.zeros((8,3))
            transform_mat[[1, 2, 5, 6]] = transform

            other_transform_mat = np.zeros((8,3))
            other_transform_mat[::] = transform
            
        elif longest_idx == 1: 
            transform = half_edges[1]
            
            transform_mat = np.zeros((8,3))
            transform_mat[[1, 2, 5, 6]] = transform

            other_transform_mat = np.zeros((8,3))
            other_transform_mat[::] = transform

        self.corner_points += transform_mat

        other_points = np.copy(self.corner_points) 
        other_points -= other_transform_mat
        other_box = bbox(other_points)

        return other_box


    def axis_align(self):
        """Axis aligns the bounding box. Calculates the minimum value against the x-axis, then rotates the box with this value. 
        """


        points_rolled = np.roll(self.corner_points, 1, axis=0)
        edges = points_rolled - self.corner_points
        
        edges_xy = edges[:, :2]

        angles = np.abs(np.arctan2(edges_xy[:, 1], edges_xy[:, 0]))
        

        
        print(np.rad2deg(angles))

        
        angle = np.rad2deg(np.min(angles))
        print(f"min angle {angle}")
        angle = angle % 90
        if angle <= 45:
            self.rotate(- angle)
        else:
            self.rotate(90-angle)
        
    def as_np_array(self):
        return self.corner_points
