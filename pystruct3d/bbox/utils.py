import numpy as np

from pystruct3d.bbox.bbox import BBox


# change to bbox array from list, pystruct3d bounding box implementation
def bbox_list2array(bbox_list: list[BBox]) -> np.ndarray:
    """Returns a numpy array of a list of bounding boxes.

    Args:
        bbox_list (list[BBox]): List of BBox of size n

    Returns:
        np.ndarray: numpy array of size nx8x3
    """

    return np.array([bbox.as_np_array() for bbox in bbox_list])


def bbox_array2list(bbox_array: np.ndarray) -> list[BBox]:
    """Return list of bounding boxes from a numpy array of shape (n, 8, 3)

    Args:
        bbox_array (np.ndarray): bounding box array, shape: (n, 8, 3)

    Returns:
        list[BBox]: List of Bbox objects
    """

    return [BBox(bbox) for bbox in bbox_array]
