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

    door_points = np.array(
        [
            [1.0, -0.2, 1.0],
            [2.0, -0.2, 1.0],
            [2.0, 0.4, 1.0],
            [1.0, 0.4, 1.0],
            [1.0, -0.2, 2.0],
            [2.0, -0.2, 2.0],
            [2.0, 0.4, 2.0],
            [1.0, 0.4, 2.0],
        ]
    )
    # pointer issues, need to copy the sample points
    i = 0
    for i in range(0, 360, 5):
        wall_box = bbox.BBox(sample_points)
        wall_box.rotate(i)
        split_box = wall_box.split_bounding_box(offset=0.5)

        wall_box.order_points()
        # print("wall box width", wall_box.width())
        # door_ref_box = bbox.BBox(door_points)
        # door_ref_box.rotate(i)
        # door_box = bbox.BBox(door_points)
        # door_box.rotate(i)

        # door_box.project_into_parent(wall_box)

        # # inliers = rand_pts[indices]

        visu = visualization.Visualization()
        # visu.bbox_geometry(door_ref_box, [0.0, 0.0, 1.0])
        visu.bbox_geometry(wall_box)
        visu.bbox_geometry(split_box, [1.0, 0.75, 0.0])
        # visu.bbox_geometry(door_box, [1.0, 0.75, 0.0])

        visu.visualize()


if __name__ == "__main__":
    main()
