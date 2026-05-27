import math

import numpy as np
from scipy.spatial import KDTree

from pystruct3d.bbox import utils


def get_centroids(bbox_list):
    bbox_array = utils.bbox_list2array(bbox_list)
    centroid = np.mean(bbox_array, axis=1)
    return centroid


def deviation(bbox_list, gt_list):
    bx_centroids = get_centroids(bbox_list)
    gt_centroids = get_centroids(gt_list)

    bx_kd_tree = KDTree(bx_centroids)

    dists = []
    for pt in gt_centroids:
        nn = bx_kd_tree.query(pt, k=1, p=1, distance_upper_bound=0.5)
        # 1st element is the distance
        if not math.isinf(nn[0]):
            dists.append(nn[0])

    return np.mean(np.asarray(dists))
    # deviation_vec = bx_centroids - gt_centroids
    # deviation = np.linalg.norm(deviation_vec, axis=1)
    # mean_deviation = np.mean(deviation)
    # print(mean_deviation)


def main():
    bx_array = np.load(
        "/home/kaufmann/scaleBIM/data/08_ShortOffice_01_F1_raw_wall_bboxes.npy"
    )
    bx_list = utils.bbox_array2list(bx_array)
    gt_array = bx_array + np.random.uniform(low=0.05, high=0.2)
    gt_list = utils.bbox_array2list(gt_array)
    deviation(bx_list, gt_list)


if __name__ == "__main__":
    main()
