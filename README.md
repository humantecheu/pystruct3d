# pystruct3d

# Installation

The `-e` flag is used for development
`pip install -e .`

# Python version

3.10

# Dependencies

Numpy

Scipy

**For visualization only**

Open3D

# Bounding box naming convention

To avoid confusion, there is a naming convention for the dimensions and points of the bounding box: 

![Bounding box naming](pystruct3d/bbox/figures/bounding_box.jpg)

Note, that the length is always the longer horizontal dimension, width is the smaller horizontal dimension. The height is the dimension along the z-axis. 

The corner points are ordered in counter-clockwise order from bottom to the top. The end points are the points of the lower horizontal center line along the length of the bounding box. 

# ToDo table

| **module**                            | **name**              | **classes, methods and functions**     | **comments**              |
|---------------------------------------|-----------------------|----------------------------------------|---------------------------|
| modified PCA axis alignment           | pca_align             | - [ ] ...                              |                           |
| voxel-based density filtering         | vox_density_filtering | - [ ] ...                              |                           |
| level fitting                         | level_fitting         | - [ ] histogram                        |                           |
|                                       |                       | - [ ] find peaks                       |                           |
|                                       |                       | - [ ] ...                              |                           |
| hypothesis based plane fitting and    | hysac                 | - [ ] fit plane svd                    |                           |
| grouping                              |                       | - [ ] seeds                            |                           |
|                                       |                       | - [ ] wall                             |                           |
|                                       |                       | - [ ] ...                              |                           |
| bounding box                          | bbox                  | - [X] *class: bbox*                    |                           |
|                                       |                       | - [X] points in bbox                   |                           |
|                                       |                       | - [X] order points                     |                           |
|                                       |                       | - [X] rotate                           |                           |
|                                       |                       | - [X] ... see code                     |                           |
|                                       |                       | - [X] fit_horizontal_bounding_box      |                           |
|                                       |                       | - [ ] ...                              |                           |
|                                       |                       | - [ ] fit_minimal_bounding_box         | experimental              |
|                                       |                       | - [ ]                                  |                           |
|                                       |                       | - [ ] ...                              |                           |
| visualization                         | visualization         | - [ ] points geometry                  |                           |
|                                       |                       | - [ ] bbox geometry                    |                           |
|                                       |                       | - [ ] plane geometry                   |                           |
|                                       |                       | - [ ] visualize geometries             |                           |
| Evalutation and metrics               | metrics               | - [ ] bbox_iou                         | hungarian algorithm, ...  |
|                                       |                       | - [ ] volumetric_iou                   |                           |
|                                       |                       | - [ ] ...                              |                           |