# Pystruct3d: all you need for bounding boxes

Pystruct3D is an open-source library to support scan-to-BIM workflows. The main purpose is to fit geometry to points of previously segmented instances, e.g., walls. For an accurate reconstruction, it is advised to apply noise filtering to instance points before any reconstruction, as reconstruction procedures might deliver bounding geometry. 

![Bounding boxes with points](docs/figures/raw_wall_bboxes_2.png)

Core features include: 
- a bounding box class with methods to fit, manipulate and get parameters of bounding boxes
- utilities to transfer point-level annotations from an annotated point cloud to data without annotations
- methods to evaluate the reconstruction accuracy by comparing reference and reconstructed geometry
- functions to visualize bounding boxes along with the points

Pystruct3d is composed of three distinct modules: (i) The bbox module, which includes a class for bounding box representation along with methods for their fitting, manipulation, and combination. (ii) The metrics module, which is equipped with methods to assess the precision of bounding box reconstructions against established \final{sets of reference bounding boxes}. (iii) The visualization module, featuring a visualizer class designed for the display of bounding boxes, individual points, and point clouds, serving as a tool for testing and diagnostic purposes. The pystruct3d module leverages high-performance libraries like [NumPy](https://github.com/numpy/numpy) and [SciPy](https://github.com/scipy/scipy) to optimize computational efficiency. In the realm of point cloud handling, [Open3D](https://github.com/isl-org/Open3D) is utilized, providing a plethora of methods and algorithms applied in this study. The visualizer class also employs [Open3D](https://github.com/isl-org/Open3D).  

To the current state, reconstruction of cuboid geometry to support building reconstruction is implemented. In future development, methods for other geometry reconstruction, e.g., to support more complex structures like bridges, will be targeted, too. 

# Installation

Clone the repository. 

From root `pystruct3d/`:
`pip install -e .`

The `-e` flag is used for development. If you only want to use the module, `pip install . ` is enough for you. 

# Python version

3.10

# Dependencies

Numpy

Scipy

Laspy

**For visualization only**

Open3D

# Bounding box naming convention

To avoid confusion, there is a naming convention for the dimensions and points of the bounding box: 

![Bounding box naming](docs/figures/bounding_box.jpg)

Note, that the length is always the longer horizontal dimension, width is the smaller horizontal dimension. The height is the dimension along the z-axis. 

The corner points are ordered in counter-clockwise order from bottom to the top. The end points are the points of the lower horizontal center line along the length of the bounding box. 

# ToDo table

- [ ] methods for bridge reconstruction

# Citation
