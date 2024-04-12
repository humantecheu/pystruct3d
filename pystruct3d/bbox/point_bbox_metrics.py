import numpy as np


def calculate_plane_normals(corner_points):
    plane_points = np.array(
        [
            corner_points[0],
            corner_points[1],
            corner_points[2],
            corner_points[3],
            corner_points[0],
            corner_points[4],
        ]
    )

    n0 = np.cross(
        corner_points[1] - plane_points[0],  # corner_points[0]
        corner_points[4] - plane_points[0],  # corner_points[0]
    )
    n0 = n0 / np.linalg.norm(n0)
    n1 = np.cross(
        corner_points[2] - plane_points[1],  # corner_points[1]
        corner_points[5] - plane_points[1],  # corner_points[1]
    )
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(
        corner_points[3] - plane_points[2],  # corner_points[2]
        corner_points[6] - plane_points[2],  # corner_points[2]
    )
    n2 = n2 / np.linalg.norm(n2)
    n3 = np.cross(
        corner_points[0] - plane_points[3],  # corner_points[3]
        corner_points[7] - plane_points[3],  # corner_points[3]
    )
    n3 = n3 / np.linalg.norm(n3)
    n4 = np.cross(
        corner_points[3] - plane_points[4],  # corner_points[0]
        corner_points[1] - plane_points[4],  # corner_points[0]
    )
    n4 = n4 / np.linalg.norm(n4)  # down
    n5 = np.cross(
        corner_points[5] - plane_points[5],  # corner_points[4]
        corner_points[7] - plane_points[5],  # corner_points[4]
    )
    n5 = n5 / np.linalg.norm(n5)  # up

    return np.array([n0, n1, n2, n3, n4, n5]), plane_points


def calculate_relative_position(
    points,
    corner_points,
    calculate_probability: bool,
):
    # Normals all point in the outward direction of the bbox faces
    plane_normals, plane_points = calculate_plane_normals(corner_points)

    # Pre-allocate memory to save time on stacking results
    dot_result = np.empty((plane_normals.shape[0], points.shape[0]))

    try:
        assert (
            np.allclose(plane_normals[0], -plane_normals[2])
            and np.allclose(plane_normals[1], -plane_normals[3])
            and np.allclose(plane_normals[4], -plane_normals[5])
        )
        # Optimized solution assumes normals of parallel faces are equal and opposite
        plane_dot = np.sum(plane_points * plane_normals, axis=1)
        point_dot = np.dot(points, plane_normals[0])
        dot_result[0] = point_dot - plane_dot[0]
        dot_result[2] = -point_dot - plane_dot[2]
        point_dot = np.dot(points, plane_normals[1])
        dot_result[1] = point_dot - plane_dot[1]
        dot_result[3] = -point_dot - plane_dot[3]
        point_dot = np.dot(points, plane_normals[4])
        dot_result[4] = point_dot - plane_dot[4]
        dot_result[5] = -point_dot - plane_dot[5]
    except AssertionError:
        print(
            "Use generalized but slower solution. Normalized normal vectors NOT equal but opposite"
        )

        """Mathematical explanation:
        dot_result = (point - plane_point) . plane_normal
                = point . plane_normal - plane_point . plane_normal
                = point_dot - plane_dot
        """

        # Generalized solution but not best computationally
        for i in range(plane_normals.shape[0]):
            dot_result[i] = np.dot(points - plane_points[i], plane_normals[i])

    # Positive dot_result is for points "above" a plane in the direction of the normal
    positive_direction = dot_result > 0

    if calculate_probability:
        return dot_result, positive_direction

    else:
        return None, positive_direction


def calculate_edge_vector(point_1, point_2):
    e = point_1 - point_2
    return e / np.linalg.norm(e)


def generate_edge_dict(corner_points):
    edges = {}
    corners = {}

    edges["e51"] = calculate_edge_vector(corner_points[5], corner_points[1])
    corners["e51"] = corner_points[1]
    edges["e62"] = calculate_edge_vector(corner_points[6], corner_points[2])
    corners["e62"] = corner_points[2]
    edges["e73"] = calculate_edge_vector(corner_points[7], corner_points[3])
    corners["e73"] = corner_points[3]
    edges["e40"] = calculate_edge_vector(corner_points[4], corner_points[0])
    corners["e40"] = corner_points[0]

    edges["e10"] = calculate_edge_vector(corner_points[1], corner_points[0])
    corners["e10"] = corner_points[0]
    edges["e54"] = calculate_edge_vector(corner_points[5], corner_points[4])
    corners["e54"] = corner_points[4]

    edges["e21"] = calculate_edge_vector(corner_points[2], corner_points[1])
    corners["e21"] = corner_points[1]
    edges["e65"] = calculate_edge_vector(corner_points[6], corner_points[5])
    corners["e65"] = corner_points[5]

    edges["e32"] = calculate_edge_vector(corner_points[3], corner_points[2])
    corners["e32"] = corner_points[2]
    edges["e76"] = calculate_edge_vector(corner_points[7], corner_points[6])
    corners["e76"] = corner_points[6]

    edges["e30"] = calculate_edge_vector(corner_points[3], corner_points[0])
    corners["e30"] = corner_points[0]
    edges["e74"] = calculate_edge_vector(corner_points[7], corner_points[4])
    corners["e74"] = corner_points[4]

    corners["c0"] = corner_points[0]
    corners["c1"] = corner_points[1]
    corners["c2"] = corner_points[2]
    corners["c3"] = corner_points[3]
    corners["c4"] = corner_points[4]
    corners["c5"] = corner_points[5]
    corners["c6"] = corner_points[6]
    corners["c7"] = corner_points[7]

    return edges, corners


def calculate_distances(
    points,
    plane_distances,
    positive_direction,
    corner_points,
):
    # Total of 27 conditions
    directions = np.array(
        [
            # Point inside bbox, condition is skipped for efficiency
            # [False, False, False, False, False, False],
            # Point closest to a plane (6 cases)
            [True, False, False, False, False, False],  # p0
            [False, True, False, False, False, False],  # p1
            [False, False, True, False, False, False],  # p2
            [False, False, False, True, False, False],  # p3
            [False, False, False, False, True, False],  # p4 # down
            [False, False, False, False, False, True],  # p5 # up
            # Point closest to a line (12 cases)
            [True, True, False, False, False, False],  # e51
            [False, True, True, False, False, False],  # e62
            [False, False, True, True, False, False],  # e73
            [True, False, False, True, False, False],  # e40
            [True, False, False, False, True, False],  # e10
            [True, False, False, False, False, True],  # e54
            [False, True, False, False, True, False],  # e21
            [False, True, False, False, False, True],  # e65
            [False, False, True, False, True, False],  # e32
            [False, False, True, False, False, True],  # e76
            [False, False, False, True, True, False],  # e30
            [False, False, False, True, False, True],  # e74
            # Point closest to a point (8 cases)
            [True, True, False, False, True, False],  # c1
            [False, True, True, False, True, False],  # c2
            [False, False, True, True, True, False],  # c3
            [True, False, False, True, True, False],  # c0
            [True, True, False, False, False, True],  # c5
            [False, True, True, False, False, True],  # c6
            [False, False, True, True, False, True],  # c7
            [True, False, False, True, False, True],  # c4
        ]
    )
    # fmt: off
    case_names = [
        "p0","p1", "p2", "p3", "p4", "p5",  # 6 plane distances
        "e51", "e62", "e73", "e40",
        "e10", "e54", "e21", "e65",
        "e32", "e76", "e30", "e74",  # 12 edge distances
        "c1", "c2", "c3", "c0", "c5", "c6", "c7", "c4",  # 8 corner distances
    ]
    # fmt: on

    shortest_distance = np.zeros((positive_direction.shape[1], 1))
    edges, corners = generate_edge_dict(corner_points)

    for i, (name, direction) in enumerate(zip(case_names, directions)):
        # exit()
        pos = (positive_direction == direction[np.newaxis].T).all(axis=0)
        if name in ["p0", "p1", "p2", "p3", "p4", "p5"]:
            shortest_distance[pos] = plane_distances[i, pos][np.newaxis].T
        # fmt: off
        elif name in ["e51", "e62", "e73", "e40", "e54", "e10",
                      "e65", "e21", "e76", "e32", "e74", "e30"]:
            # fmt: on
            shortest_distance[pos] = np.linalg.norm(
                np.cross(points[pos] - corners[name], edges[name]),
                axis=1,
            )[np.newaxis].T
        elif name in ["c1", "c2", "c3", "c0", "c5", "c6", "c7", "c4"]:
            shortest_distance[pos] = np.linalg.norm(
                points[pos] - corners[name],
                axis=1,
            )[np.newaxis].T
        else:
            print("Weird behaviour, debug")

    return shortest_distance


# def asvoid(arr):
#     """
#     Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
#     View the array as dtype np.void (bytes). The items along the last axis are
#     viewed as one value. This allows comparisons to be performed which treat
#     entire rows as one value.
#     """
#     arr = np.ascontiguousarray(arr)
#     if np.issubdtype(arr.dtype, np.floating):
#         """Care needs to be taken here since
#         np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
#         Adding 0. converts -0. to 0.
#         """
#         arr += 0.0
#     return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))
