# pystruct3d


# Python version

3.10

# Dependencies

Numpy

Scipy

**For visualization only**

Open3D

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
| bounding box                          | bbox                  | - [ ] *class: bbox*                    |                           |
|                                       |                       | - [ ] points in bbox                   |                           |
|                                       |                       | - [ ] order points                     |                           |
|                                       |                       | - [ ] rotate                           |                           |
|                                       |                       | - [ ] *class: hobb*                    |                           |
|                                       |                       | - [ ] fit                              |                           |
|                                       |                       | - [ ] ...                              |                           |
|                                       |                       | - [ ] *class: min_bbox*                | experimental              |
|                                       |                       | - [ ] fit                              |                           |
|                                       |                       | - [ ] ...                              |                           |
| visualization                         | visualization         | - [ ] points geometry                  |                           |
|                                       |                       | - [ ] bbox geometry                    |                           |
|                                       |                       | - [ ] plane geometry                   |                           |
|                                       |                       | - [ ] visualize geometries             |                           |
| Evalutation and metrics               | metrics               | - [ ] bbox_iou                         | hungarian algorithm, ...  |
|                                       |                       | - [ ] volumetric_iou                   |                           |
|                                       |                       | - [ ] ...                              |                           |