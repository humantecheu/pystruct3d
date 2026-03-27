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

ref_box = bbox.BBox(sample_points)

start = time.time()
pts_1, idx_1, _ = ref_box.points_in_bbox_probability(rand_pts, 0.7)
print(time.time() - start)

visu = visualization.Visualization()
visu.bbox_geometry(ref_box)
visu.point_cloud_geometry(pts_1)
visu.visualize()
