import numpy as np
import pytest

from pystruct3d.testing import create_bbox_lists, generate_bounding_boxes


@pytest.fixture
def bbox_pair():
    """Paired (gt, pred) bounding box lists — 20 GT boxes, ~16 slightly shifted predictions."""
    return create_bbox_lists()


@pytest.fixture
def small_bbox_list():
    """5 axis-aligned synthetic boxes for lightweight BBox method tests."""
    return generate_bounding_boxes(5)


def axis_box(
    x: float, y: float, z: float, dx: float, dy: float, dz: float
) -> np.ndarray:
    """(8, 3) axis-aligned box corners from origin + extents."""
    return np.array(
        [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ],
        dtype=np.float64,
    )
