import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.bbox.utils import bbox_list2array


def pointcloud_limits(pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        pointcloud (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    assert pointcloud.shape[1] == 3, "pointcloud must be of shape nx3"
    min_values = np.floor(np.min(pointcloud, axis=0)).astype(int)
    max_values = np.ceil(np.max(pointcloud, axis=0)).astype(int)

    # Return the two diagonal points
    return min_values, max_values


def bbox_limits(bboxes_list: list[BBox]) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        bboxes_list (list[BBox]): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    bbox_array = bbox_list2array(bboxes_list)

    min_vals = np.floor(np.min(bbox_array.reshape(-1, 3), axis=0)).astype(int)
    max_vals = np.ceil(np.max(bbox_array.reshape(-1, 3), axis=0)).astype(int)

    return min_vals, max_vals


def voxelization_limits(
    *args: np.ndarray | BBox | list[BBox],
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    limits_list = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            limits_list.append(pointcloud_limits(arg))
        elif isinstance(arg, BBox):
            limits_list.append(bbox_limits([arg]))
        elif isinstance(arg, list) and all(isinstance(item, BBox) for item in arg):
            limits_list.append(bbox_limits(arg))
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
