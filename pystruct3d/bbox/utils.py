import numpy as np
from . import bbox


# change to bbox array from list, pystruct3d bounding box implementation
def bbox_array_from_list(bboxes: list[bbox.BBox]) -> np.ndarray:
    """Return bounding box array from list of bounding boxes

    Args:
        wall_hBboxes (list): list of hBboxes

    Returns:
        np.ndarray: bounding box array, shape: (n, 8, 3)
    """

    bbox_array = np.empty((0, 8, 3))
    for bbx in bboxes:
        bbox_points = bbx.corner_points
        bbox_array = np.append(bbox_array, bbox_points.reshape((1, 8, 3)), axis=0)

    return bbox_array


def bbox_list_from_array(bboxes: np.ndarray) -> list[bbox.BBox]:
    """Return list of Bboxes from np.ndarray of shape (n, 8, 3)

    Args:
        bboxes (np.ndarray): bounding box array, shape: (n, 8, 3)

    Returns:
        list[bbox.BBox]: List of Bbox objects
    """
    bbox_list = []
    for bx in bboxes:
        box = bbox.BBox(bx)
        bbox_list.append(box)
    return bbox_list
