# pystruct3d

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

![Bounding box naming](pystruct3d/bbox/figures/bounding_box.jpg)

Note, that the length is always the longer horizontal dimension, width is the smaller horizontal dimension. The height is the dimension along the z-axis. 

The corner points are ordered in counter-clockwise order from bottom to the top. The end points are the points of the lower horizontal center line along the length of the bounding box. 

# ToDo table

- [X] change points_in_BBox to points_in_bbox
- [X] implement from_cv4aec bbox method

| **module**                            | **name**              | **classes, methods and functions**     | **comments**              |
|---------------------------------------|-----------------------|----------------------------------------|---------------------------|
| modified PCA axis alignment           | pca_align             | - [ ] ...                              |                           |
| voxel-based density filtering         | vox_density_filtering | - [ ] ...                              |                           |
| level fitting                         | level_fitting         | - [ ] histogram                        |                           |
|                                       |                       | - [ ] find peaks                       |                           |
|                                       |                       | - [ ] ...                              |                           |
| bounding box                          | bbox                  | - [X] *class: bbox*                    |                           |
|                                       |                       | - [X] order_points                     |                           |
|                                       |                       | - [X] points_in_bbox                   |                           |
|                                       |                       | - [X] translate                        |                           |
|                                       |                       | - [X] translate_z                      |                           |
|                                       |                       | - [X] expand                           |                           |
|                                       |                       | - [X] lower_edges                      |                           |
|                                       |                       | - [X] length                           |                           |
|                                       |                       | - [X] width                            |                           |
|                                       |                       | - [X] height                           |                           |
|                                       |                       | - [X] angle                            |                           |
|                                       |                       | - [X] split_bounding_box               |                           |
|                                       |                       | - [X] axis_align                       |                           |
|                                       |                       | - [X] as_np_array                      |                           |
|                                       |                       | - [X] get_endpts                       |                           |
|                                       |                       | - [X] volume                           |                           |
|                                       |                       | - [X] fit_axis_aligned                 |                           |
|                                       |                       | - [X] fit_horizontal_aligned           |                           |
|                                       |                       | - [ ] fit_minimal                      |                           |
|                                       |                       | - [X] bbox_from_verts                  |                           |
|                                       |                       | - [ ] ...                              |                           |
| visualization                         | visualization         | - [X] o3d_point_cloud                  |                           |
|                                       |                       | - [X] point_cloud_geometry             |                           |
|                                       |                       | - [X] bbox_geometry                    |                           |
|                                       |                       | - [ ] plane_geometry                   |                           |
|                                       |                       | - [X] points_geometry                  |                           |
|                                       |                       | - [X] visualize                        |                           |
| Evalutation and metrics               | metrics               | - [X] bbox_iou                         | hungarian algorithm, ...  |
|                                       |                       | - [X] volumetric_iou                   |                           |
|                                       |                       | - [X] voxel_iou                        |                           |
|                                       |                       | - [ ] ...                              |                           |