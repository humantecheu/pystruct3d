import time

import numpy as np

from pystruct3d.bbox import bbox
from pystruct3d.visualization import visualization

rand_pts = np.random.uniform(-100, 100, size=(100000000, 3))

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

# test_points = np.array([[2.5, 0.5, 1.5], [2.5, -0.5, 1.5]])
# ref_box.rotate(45)
# other_bx = ref_box.split_bounding_box()
# bx.axis_align()

ref_box = bbox.BBox(sample_points)

start = time.time()
pts_1, idx_1, _ = ref_box.points_in_bbox_probability(rand_pts, 0.7)
print(time.time() - start)
# exit()
# print(len(idx_1))
# start = time.time()
# pts_2, idx_2 = ref_box.points_in_BBox(rand_pts)
# print(time.time() - start)
# print(len(idx_2))
# print(np.array_equal(pts_1, pts_2))
# print(np.array_equal(np.sort(idx_1), np.sort(idx_2)))
# exit()


visu = visualization.Visualization()

visu.bbox_geometry(ref_box)
# visu.points_geometry(test_points)
visu.point_cloud_geometry(pts_1)

visu.visualize()
