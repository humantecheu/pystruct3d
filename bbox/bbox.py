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
    
    def points_in_bbox(self, points, tolerance=1e-12):
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
        
    def as_np_array(self):
        return self.corner_points
    

    

if __name__=="__main__":

    rand_pts = np.random.uniform(0, 6, size=(200000,3))
    sample_points = np.array([
        [0., 0., 0.], 
        [5., 0., 0.], 
        [5., 1., 0.],
        [0., 1., 0.],
        [0., 0., 3.],
        [5., 0., 3.],
        [5., 1., 3.],
        [0., 1., 3.]
    ])
    print(sample_points.shape)
    bx = bbox(sample_points)
    print(rand_pts)
    # bx.rotate(45)
    points_in, idx = bx.points_in_bbox(rand_pts)
    print(points_in)
    print(np.shape(idx))
    box_array = bx.as_np_array()
    print(f"Box array shape: {box_array.shape}")