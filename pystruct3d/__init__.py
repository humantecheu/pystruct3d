# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
"""pystruct3d — bounding box fitting, evaluation metrics, and visualization for scan-to-BIM.

Modules:

- **bbox**: `BBox` class — fitting, geometric properties, containment, transforms, serialization
- **io**: point cloud readers (LAS/LAZ, E57, PCD/PLY/XYZ) and CV4AEC format I/O
- **metrics**: BBox IoU, volumetric IoU, voxel IoU, precision/recall, centroid deviation
- **preprocessing**: alignment, voxel downsampling, PCA, cropping, label filtering
- **annotation**: nearest-neighbour label transfer between point clouds
- **visualization**: interactive Open3D viewer via `Visualizer`
- **testing**: synthetic BBox generators for benchmarks and quick demos

Quickstart::

    import numpy as np
    from pystruct3d.bbox import BBox
    from pystruct3d.io import read_point_cloud

    xyz, rgb = read_point_cloud("scan.las")
    box = BBox()
    box.fit_horizontal_aligned(xyz)
    print(box.length(), box.width(), box.height())

Logging:

pystruct3d registers a `NullHandler` — silent by default. Enable with::

    import logging
    logging.basicConfig(level=logging.INFO)

Pass ``progress=False`` to any loader or ``transfer_labels`` to suppress tqdm bars.
"""

import logging
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

logging.getLogger(__name__).addHandler(logging.NullHandler())

with suppress(PackageNotFoundError):
    __version__ = version("pystruct3d")
