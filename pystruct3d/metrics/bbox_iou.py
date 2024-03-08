from typing import List

import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.metrics.generate_example import create_bbox_lists
from pystruct3d.metrics.munkres import Munkres
from pystruct3d.metrics.voxelization_limits import bbox_list2array
from pystruct3d.visualization.visualization import Visualization


def line_intersection_2d(line_1: np.ndarray, line_2: np.ndarray) -> np.ndarray:
    # Each line is defined by two points: (x1, y1, z1) and (x2, y2, z2)
    # Extract the x and y coordinates of the points
    x1, y1, _ = line_1[0]
    x2, y2, _ = line_1[1]
    x3, y3, _ = line_2[0]
    x4, y4, _ = line_2[1]

    # Calculate the coefficients for the lines
    # Line1: (a1 * x) + (b1 * y) = c1
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    # Line2: (a2 * x) + (b2 * y) = c2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    # Calculate the determinant
    det = a1 * b2 - a2 * b1

    if det == 0:
        # The lines are parallel or coincident
        return None
    else:
        # Calculate the intersection point
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return np.array([x, y])


def on_segment(line: np.ndarray, point: np.ndarray) -> bool:
    """Check if point c is between points a and b."""
    a = line[0, :2]
    b = line[1, :2]
    crossproduct = (point[1] - a[1]) * (b[0] - a[0]) - (point[0] - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > 1e-10:  # Allow a small error margin
        return False

    dotproduct = (point[0] - a[0]) * (b[0] - a[0]) + (point[1] - a[1]) * (b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    if dotproduct > squaredlengthba:
        return False

    return True


def bbox_2d_intersection(bbox_1: BBox, bbox_2: BBox):
    bbox_1_2d = bbox_1.corner_points[:4]
    bbox_2_2d = bbox_2.corner_points[:4]

    vertices_2d = []

    for i in range(bbox_1_2d.shape[0]):
        for j in range(bbox_2_2d.shape[0]):
            line_1 = np.array([bbox_1_2d[i], bbox_1_2d[i - 1]])
            line_2 = np.array([bbox_2_2d[j], bbox_2_2d[j - 1]])
            intersection = line_intersection_2d(line_1, line_2)
            # if there is an intersection check that it belongs to the segments
            if (
                intersection is not None
                and on_segment(line_1, intersection)
                and on_segment(line_2, intersection)
            ):
                vertices_2d.append(intersection)

            # if there is no intersection check if they are colinear
            elif on_segment(line_1, bbox_2_2d[j]):
                vertices_2d.append(bbox_2_2d[j, :2])

            elif on_segment(line_2, bbox_1_2d[i]):
                vertices_2d.append(bbox_1_2d[i, :2])

    vertices_2d = np.array(vertices_2d)
    points, _ = bbox_1.points_in_bbox_probability(bbox_2_2d, 0, True)
    if points.any():
        vertices_2d = np.vstack((vertices_2d, points[:, :2]))
    points, _ = bbox_2.points_in_bbox_probability(bbox_1_2d, 0, True)
    if points.any():
        vertices_2d = np.vstack((vertices_2d, points[:, :2]))

    # Only return unique 2D points
    vertices_2d = np.unique(vertices_2d.round(decimals=4), axis=0)
    return vertices_2d


def bbox_z_intersection(bbox_1: BBox, bbox_2: BBox):
    min_1 = bbox_1.corner_points[0, 2]
    max_1 = bbox_1.corner_points[4, 2]
    min_2 = bbox_2.corner_points[0, 2]
    max_2 = bbox_2.corner_points[4, 2]

    # Check if there's an overlap
    if max(min_1, min_2) <= min(max_1, max_2):
        return np.array([max(min_1, min_2), min(max_1, max_2)])
    else:
        return None  # No intersection


def shoelace_2d_area(vertices: np.ndarray):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def bbox_intersection_volume(intersection_2d: np.ndarray, intersection_z: np.ndarray):
    def ccw_sort(vertices: np.ndarray):
        centroid = np.mean(vertices, axis=0)
        sorted_vertices = vertices[
            np.argsort(
                np.arctan2(
                    vertices[:, 1] - centroid[1],
                    vertices[:, 0] - centroid[0],
                )
            ),
            :,
        ]
        return sorted_vertices

    if intersection_2d.shape[0] < 3:
        # If less than 3 vertices no area exists
        return 0

    area_2d = shoelace_2d_area(ccw_sort(intersection_2d))
    z_range = np.abs(intersection_z[1] - intersection_z[0])
    return area_2d * z_range


def bbox_iou(bbox_1: BBox, bbox_2: BBox):
    intersection_z = bbox_z_intersection(bbox_1, bbox_2)
    intersection_volume = 0
    union_volume = 1
    if intersection_z is not None:
        intersection_2d = bbox_2d_intersection(bbox_1, bbox_2)
        intersection_volume = bbox_intersection_volume(intersection_2d, intersection_z)
        union_volume = bbox_1.volume() + bbox_2.volume() - intersection_volume

    return intersection_volume / union_volume


def mean_bbox_iou(
    groundtruth_bbox_list: List[BBox],
    predicted_bbox_list: List[BBox],
) -> float:
    """_summary_

    Args:
        groundtruth_bbox_list (List[BBox]): _description_
        predicted_bbox_list (List[BBox]): _description_

    Returns:
        _type_: _description_
    """
    dim = max(len(groundtruth_bbox_list), len(predicted_bbox_list))
    iou_matrix = np.zeros((dim, dim))
    for i, gt_bbox in enumerate(groundtruth_bbox_list):
        for j, pd_bbox in enumerate(predicted_bbox_list):
            iou_matrix[i, j] = bbox_iou(gt_bbox, pd_bbox)

    munkres = Munkres()
    idx = munkres.compute(iou_matrix, maximize=True)
    ious = iou_matrix[idx[:, 0], idx[:, 1]]

    return np.mean(ious)


def main() -> None:
    gt_boxes, pd_boxes = create_bbox_lists()
    visualizer = Visualization()
    for box in gt_boxes:
        visualizer.bbox_geometry(box, [1, 0, 0])
    for box in pd_boxes:
        visualizer.bbox_geometry(box, [0, 0, 1])

    visualizer.visualize()

    print(mean_bbox_iou(gt_boxes, pd_boxes))


if __name__ == "__main__":
    main()
