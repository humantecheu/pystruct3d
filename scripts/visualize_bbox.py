import time

import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.visualization import Visualizer

rand_pts = np.random.uniform(-100, 100, size=(100_000_000, 3))

sample_points = np.array([
    [0.0, 0.0, 0.0],
    [5.0, 0.0, 0.0],
    [5.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 3.0],
    [5.0, 0.0, 3.0],
    [5.0, 1.0, 3.0],
    [0.0, 1.0, 3.0],
])

ref_box = BBox(sample_points)

start = time.time()
pts_1, idx_1, _ = ref_box.points_in_bbox_soft(rand_pts, 0.7)
print(time.time() - start)

Visualizer().add_bbox(ref_box).add_points(pts_1).show()
