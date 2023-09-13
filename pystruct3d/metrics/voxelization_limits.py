from typing import List, Tuple

import numpy as np

from pystruct3d.bbox.bbox import BBox


def bbox_list2array(bbox_list: List[BBox]) -> np.ndarray:
    """Returns a numpy array of a list of bounding boxes.

    Args:
        bbox_list (List[BBox]): List of BBox of size n

    Returns:
        np.ndarray: numpy array of size nx8x3
    """

    return np.array([bbox.as_np_array() for bbox in bbox_list])


def pc_limits(pointcloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        pointcloud (np.ndarray): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    assert pointcloud.shape[1] == 3, "pointcloud must be of shape nx3"
    min_values = np.floor(np.min(pointcloud, axis=0)).astype(int)
    max_values = np.ceil(np.max(pointcloud, axis=0)).astype(int)

    # Return the two diagonal points
    return min_values, max_values


def bbox_limits(bboxes_list: List[BBox]) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        bboxes_list (List[BBox]): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    bbox_array = bbox_list2array(bboxes_list)

    min_vals = np.floor(np.min(bbox_array.reshape(-1, 3), axis=0)).astype(int)
    max_vals = np.ceil(np.max(bbox_array.reshape(-1, 3), axis=0)).astype(int)

    return min_vals, max_vals


def voxelization_limits(
    *args: np.ndarray | BBox | List[BBox],
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    limits_list = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            limits_list.append(pc_limits(arg))
        elif isinstance(arg, List) and all(isinstance(item, BBox) for item in arg):
            limits_list.append(bbox_limits(arg))
        elif isinstance(arg, BBox):
            limits_list.append(bbox_limits([arg]))
        else:
            raise TypeError("Arguments can either be of type np.ndarray or List[BBox]")

    # Split the limits into min and max points
    min_list, max_list = zip(*limits_list)
    min_array = np.array(min_list)
    max_array = np.array(max_list)

    # Find overall min and max
    limits_min = np.min(min_array, axis=0).astype(int)
    limits_max = np.max(max_array, axis=0).astype(int)

    return limits_min, limits_max
