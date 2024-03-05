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

    bx = bbox.BBox(sample_points)

    bx.get_center_plane()

    visu = visualization.Visualization()
    visu.bbox_geometry([bx])

    visu.visualize()


if __name__ == "__main__":
    main()
