import numpy as np

from pystruct3d.bbox.bbox import BBox


def create_bounding_box(
    bottom_corner: tuple[float, float, float],
    l: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Create a bounding box with bottom left corner at (x, y, z),
    length l, width w, and height h

    Args:
        bottom_corner (tuple[float, float, float]): bottom left corner
        l (float): length (m)
        w (float): width (m)
        h (float): height (m)

    Returns:
        np.ndarray: 8x3 array of bounding box corners
    """
    x, y, z = bottom_corner
    return np.array(
        [
            [x, y, z],
            [x + l, y, z],
            [x + l, y + w, z],
            [x, y + w, z],
            [x, y, z + h],
            [x + l, y, z + h],
            [x + l, y + w, z + h],
            [x, y + w, z + h],
        ]
    )


def generate_bounding_boxes(
    n_boxes: int = 15,
    l_range: tuple[float, float] = (3, 5),
    w_range: tuple[float, float] = (1, 2),
    h_range: tuple[float, float] = (2, 3),
) -> list[BBox]:
    """Generate n_boxes bounding boxes with random sizes and positions

    Args:
        n_boxes (int, optional): _description_. Defaults to 15.
        l_range (tuple[float, float], optional): _description_. Defaults to (3, 5).
        w_range (tuple[float, float], optional): _description_. Defaults to (1, 2).
        h_range (tuple[float, float], optional): _description_. Defaults to (2, 3).

    Returns:
        list[BBox]: _description_
    """
    assert n_boxes > 1, f"n_boxes must be greater than or equal to 1!"
    x_range = (0, n_boxes)
    y_range = (0, n_boxes)
    z_range = (0, 2 * np.array(h_range).max())

    bounding_boxes = []
    x, y = 0, 0
    max_w = 0
    for _ in range(n_boxes):
        # Generate random dimensions
        l = np.random.uniform(*l_range)
        w = np.random.uniform(*w_range)
        h = np.random.uniform(*h_range)
        z = np.random.uniform(*z_range)

        # Create bounding box
        box = create_bounding_box((x, y, z), l, w, h)
        bounding_boxes.append(BBox(box))

        # Move to the next position
        x += l + 1 + np.random.uniform(0, 1)
        if x > x_range[1]:
            x = 0
            y += max_w + 1 + np.random.uniform(0, 1)
        max_w = max(max_w, w)

        # Check if we've moved out of the y-range
        if y > y_range[1]:
            break

    return bounding_boxes


def shift_bounding_boxes(
    bounding_boxes: list[BBox],
    n_boxes_to_shift: float,
    shift_range: tuple[float, float] = (-1, 1),
) -> list[BBox]:
    """Randomly select a subset of bounding boxes and shift them
    randomly in a small region around their original position

    Args:
        bounding_boxes (list[BBox]): _description_
        n_boxes_to_shift (float): _description_
        shift_range (tuple[float, float], optional): _description_. Defaults to (-1, 1).

    Returns:
        list[BBox]: _description_
    """
    shifted_boxes = []

    # Randomly select boxes to shift
    indices_to_shift = np.random.choice(
        len(bounding_boxes), size=n_boxes_to_shift, replace=False
    )

    for i in indices_to_shift:
        # Generate random shift
        shift = np.random.uniform(shift_range[0], shift_range[1], size=3)

        # Apply shift to box
        shifted_boxes.append(BBox(bounding_boxes[i].as_np_array() + shift))

    return shifted_boxes


def create_bbox_lists() -> tuple[list[BBox], list[BBox]]:
    # Generate bounding boxes
    bboxes_1 = generate_bounding_boxes(20)

    n_boxes_to_shift = np.ceil(len(bboxes_1) * 0.8).astype(int)
    # Shift some bounding boxes
    bboxes_2 = shift_bounding_boxes(bboxes_1, n_boxes_to_shift)

    # print(len(bboxes_1), len(bboxes_2))

    return bboxes_1, bboxes_2
