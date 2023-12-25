import numpy as np


# change to bbox array from list, pystruct3d bounding box implementation
def bbox_array_from_list(wall_hBboxes):
    """return bounding box array from list of bounding boxes

    Args:
        wall_hBboxes (list): list of hBboxes

    Returns:
        np.ndarray: bounding box array, shape: (n, 8, 3)
    """

    bbox_array = np.empty((0, 8, 3))
    for bbx in wall_hBboxes:
        bbox_points = bbx.corner_points
        bbox_array = np.append(bbox_array, bbox_points.reshape((1, 8, 3)), axis=0)

    return bbox_array
