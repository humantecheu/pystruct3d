from typing import List

import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.metrics.generate_example import create_bbox_lists
from pystruct3d.metrics.voxelization_limits import bbox_list2array
from pystruct3d.visualization.visualization import Visualization


def bbox_iou(groundtruth_bbox_list: List[BBox], predicted_bbox_list: List[BBox]):
    """_summary_

    Args:
        groundtruth_bbox_list (List[bbox]): _description_
        predicted_bbox_list (List[bbox]): _description_
    """
    groundtruth_array = bbox_list2array(groundtruth_bbox_list)
    predicted_array = bbox_list2array(predicted_bbox_list)

    # minimal = np.min([groundtruth_array.min(), predicted_array.min(), 0.0])
    # groundtruth_array -= minimal
    # predicted_array -= minimal


def mean_bbox_iou():
    pass


def main() -> None:
    gt_boxes, pd_boxes = create_bbox_lists()
    visualizer = Visualization()
    for box in gt_boxes:
        visualizer.bbox_geometry(box, [1, 0, 0])
    for box in pd_boxes:
        visualizer.bbox_geometry(box, [0, 0, 1])

    visualizer.visualize()

    bbox_iou(gt_boxes, pd_boxes)


if __name__ == "__main__":
    main()
