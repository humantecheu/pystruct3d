import numpy as np

from pystruct3d.visualization import visualization
from pystruct3d.bbox import bbox


def main():
    rand_pts = np.random.uniform(-10, 10, size=(2000000, 3))

    sample_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [5.0, 0.0, 3.0],
            [5.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ]
    )
    # pointer issues, need to copy the sample points

    ref_box = bbox.BBox(sample_points)
    inliers, indices = ref_box.points_in_bbox(rand_pts)

    # inliers = rand_pts[indices]

    visu = visualization.Visualization()

    visu.bbox_geometry(ref_box)

    visu.point_cloud_geometry(inliers)

    visu.visualize()


if __name__ == "__main__":
    main()
