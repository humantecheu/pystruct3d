import numpy as np
from scipy.spatial import KDTree
import math

from pystruct3d.bbox.bbox import BBox
from pystruct3d.bbox import utils


def get_centroids(bbox_list):
    bbox_array = utils.bbox_array_from_list(bbox_list)
    centroid = np.mean(bbox_array, axis=1)
    return centroid


def deviation(bbox_list, gt_list):
    bx_centroids = get_centroids(bbox_list)
    gt_centroids = get_centroids(gt_list)
    print(bx_centroids, gt_centroids)

    bx_kd_tree = KDTree(bx_centroids)
    gt_kd_tree = KDTree(gt_centroids)

    dists = []
    for pt in gt_centroids:
        nn = bx_kd_tree.query(pt, k=1, p=1, distance_upper_bound=0.5)
        # 1st element is the distance
        if not math.isinf(nn[0]):
            dists.append(nn[0])
    print(dists)
    print(f"Centroid deviation {np.mean(np.asarray(dists))}")
    return np.mean(np.asarray(dists))
    # deviation_vec = bx_centroids - gt_centroids
    # deviation = np.linalg.norm(deviation_vec, axis=1)
    # mean_deviation = np.mean(deviation)
    # print(mean_deviation)


def main():
    # arr = np.load(
    #     "/home/kaufmann/scaleBIM/data/08_ShortOffice_01_F1_raw_wall_bboxes.npy"
    # )
    bx1 = np.array(
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
    bx1_gt = bx1 + 0.2
    bx2 = np.array(
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
    bx2_gt = bx2 + 0.2
    # get_centroids(arr)
    # bx_list = [BBox(bx1), BBox(bx2)]
    # gt_bx_list = [BBox(bx1_gt), BBox(bx2_gt)]
    # deviation(bx_list, gt_bx_list)
    bx_array = np.load(
        "/home/kaufmann/scaleBIM/data/08_ShortOffice_01_F1_raw_wall_bboxes.npy"
    )
    bx_list = utils.bbox_list_from_array(bx_array)
    gt_array = bx_array + np.random.uniform(low=0.05, high=0.2)
    gt_list = utils.bbox_list_from_array(gt_array)
    deviation(bx_list, gt_list)


if __name__ == "__main__":
    main()
