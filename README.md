# pystruct3d

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2604.24311-b31b1b.svg)](https://arxiv.org/abs/2604.24311)

pystruct3d is an open-source Python library supporting scan-to-BIM workflows as part of the [BIMStruct3D](https://arxiv.org/abs/2604.24311) pipeline. It provides tools for fitting bounding geometry to segmented point cloud instances, evaluating reconstruction accuracy, and visualizing 3D structural elements. For accurate reconstruction it is advised to apply noise filtering to instance points beforehand, as reconstruction procedures may produce bounding geometry.

![Bounding boxes with points](docs/figures/wall_bbox_reconstruction.png)

## Features

- **bbox**: Bounding box class with methods for fitting, manipulating, and querying box parameters from point clouds
- **metrics**: Evaluation metrics for comparing reconstructed against reference geometry, including volumetric IoU (vIoU) for instance-free reconstruction assessment
- **visualization**: Visualizer class for displaying bounding boxes, points, and point clouds
- **annotation**: Utilities to transfer point-level annotations from an annotated point cloud to unannotated data
- **preprocessing**: Point cloud preprocessing including axis alignment and array search utilities

pystruct3d leverages [NumPy](https://github.com/numpy/numpy) and [SciPy](https://github.com/scipy/scipy) for computational efficiency, and [Open3D](https://github.com/isl-org/Open3D) for point cloud handling and visualization.

## Installation

Clone the repository and install from the root `pystruct3d/` directory:

```shell
pip install -e .
```

The `-e` flag is for development. For use only, `pip install .` is sufficient.

## Requirements

- Python 3.12+
- NumPy
- SciPy
- laspy
- pye57

**Visualization only:**
- Open3D

## Bounding box naming convention

To avoid confusion, there is a naming convention for the dimensions and points of the bounding box:

![Bounding box naming](docs/figures/bounding_box.png)

The length is always the longer horizontal dimension, width is the smaller horizontal dimension, and height is the dimension along the z-axis. Corner points are ordered counter-clockwise from bottom to top. End points are the lower horizontal center line points along the length of the bounding box.

## Related Resources

- **Paper**: [BIMStruct3D: A Fully Automated Hybrid Learning Scan-to-BIM Pipeline with Integrated Topology Refinement](https://arxiv.org/abs/2604.24311)
- **Dataset**: [DeKH — German Hospital Dataset](https://huggingface.co/datasets/RPTU-FGMB/DeKH) — annotated point clouds and ground-truth IFC BIM models used in the paper

## Citing this Work

```bibtex
@article{chamseddine2026bimstruct3d,
    title   = {BIMStruct3D: A Fully Automated Hybrid Learning Scan-to-BIM Pipeline with Integrated Topology Refinement},
    author  = {Chamseddine, Mahdi and Kaufmann, Fabian and Schellen, Marius and Glock, Christian and Stricker, Didier and Rambach, Jason},
    journal = {arXiv preprint arXiv:2604.24311},
    year    = {2026}
}
```

## History

**2026-05-28 — history rewrite for author attribution**

Two early commits were authored under an incorrect name and email address. The repository history was rewritten on this date to correct the attribution. If you cloned this repository before 2026-05-28, your local clone will have diverged history. Re-clone to get the corrected history:

```bash
git clone https://github.com/humantecheu/pystruct3d.git
```

## Acknowledgement

This research was funded by the European Union as part of the projects: HumanTech (Grant Agreement 101058236) and ShieldBOT (Grant Agreement 101235093).

## License

MIT License. See [LICENSE](LICENSE) for details.
