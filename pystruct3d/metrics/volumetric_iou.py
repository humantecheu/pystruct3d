from typing import Tuple

import numpy as np

from pystruct3d.bbox.bbox import BBox
from pystruct3d.metrics.generate_example import create_bbox_lists


def voxelize_bbox(
    bbox: BBox,
    volume_limits: Tuple[np.ndarray, np.ndarray],
    voxel_size: float,
):
    """_summary_

    Args:
        bbox (BBox): _description_
        volume_limits (Tuple[np.ndarray, np.ndarray]): _description_
        voxel_size (float): _description_
    """
    pass


def main():
    pass


if __name__ == "__main__":
    main()
