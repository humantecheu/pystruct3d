# Changelog

All notable changes to pystruct3d are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.8.0] - 2026-05-28

### Added
- `io.e57`: `read_e57_file` reads all scans via pye57, concatenates them, and normalises colors with the same 8/16-bit detection logic as the LAS reader
- `io.readers`: `read_point_cloud` dispatcher routes by file extension — LAS/LAZ via laspy, E57 via pye57, PCD/PLY/XYZ/PTS via open3d

### Changed (Breaking)
- `io.las.read_las_file`: now accepts `Path | str` and returns `(xyz, rgb)` as `tuple[np.ndarray, np.ndarray]` instead of `o3d.geometry.PointCloud`; RGB bit-depth (8 vs 16-bit) is detected automatically
- `io.las.write_las_file`: now accepts `(las_path, xyz, rgb)` numpy arrays instead of an open3d PointCloud

## [0.7.1] - 2026-05-28

### Added
- `preprocessing.labels`: `filter_ids` and `labels_to_color` utilities for working with per-point semantic and instance label arrays

## [0.7.0] - 2026-05-28

### Changed
- `visualization`: rewritten as a fluent `Visualizer` builder — `add_bbox`, `add_points`, `add_point_cloud`, `add_markers`, `add_coordinate_frame`, `remove`, `clear`, `show`, `show_with_animation` can all be chained

## [0.6.1] - 2026-05-28

### Added
- `testing` module with shared test helpers
- `BBox.from_params`: consolidated origin modes (centroid vs corner)

## [0.6.0] - 2026-05-28

### Changed
- `utils` merged into `io`; all public symbols re-exported from `pystruct3d.io`
- `BBox` cv4aec I/O methods deprecated in favour of `io.cv4aec`

## [0.5.0] - 2026-05-27

### Added
- `BBox.to_dict` / `BBox.from_dict` serialization
- `io.cv4aec`: `bbox_from_cv4aec` and `bbox_to_cv4aec` for CV4AEC format I/O
- KD-tree pruning in `mean_bbox_iou` for scalable nearest-neighbour assignment

## [0.4.2] - 2026-05-27

### Changed
- `BBox.fit_horizontal_aligned`: vectorised implementation, 4–8× faster than the previous loop-based version

## [0.4.1] - 2026-05-27

### Added
- Open3D signed-distance-field (SDF) soft containment test for `BBox`
- 2D slab bounding box test

### Removed
- Orphaned internal modules that had no public API

## [0.4.0] - 2026-05-27

### Added
- `BBox.to_o3d` / `BBox.from_o3d`: interop with open3d `OrientedBoundingBox`
- Open3D OBB containment check
- SciPy linear assignment (`scipy.optimize.linear_sum_assignment`) as an alternative solver alongside Munkres

## [0.3.1] - 2026-05-27

### Added
- Hot-path benchmarks for bbox fitting and metrics

### Removed
- Stray scripts and debug files that were accidentally included in the package tree

## [0.3.0] - 2026-05-27

### Changed
- Replaced unmaintained `e57` dependency with `pye57`; added `read_e57_as_point_cloud` helper in `annotation.utils`
- `BBox.__init__`: fixed mutable default argument (numpy sentinel replaced with `None` guard)
- `BBox.points_in_bbox`: replaced bare `except` with `except (QhullError, ValueError)`
- `BBox.fit_minimal`: added missing `self`, raises `NotImplementedError` instead of silently passing
- Replaced all debug `print()` calls with `warnings.warn`
- Removed dead code in `metrics.centroid_deviation`

## [0.2.2] - 2026-05-19

### Added
- Wall point cloud visualisation figure as README hero image

## [0.2.1] - 2026-05-19

### Added
- Bounding box naming convention diagram (`docs/figures/bounding_box.png`)
- arXiv, license, and Python version badges
- Related Resources and Citation sections in README

## [0.2.0] - 2026-03-27

### Added
- `preprocessing` module: `alignment.py` (`align_to_axes`), `array_row_search.py`
- `metrics.hungarian_algorithm`: pure-NumPy Hungarian/LAP solver as alternative to Munkres
- `scripts/visualize_bbox.py` utility

### Changed
- Type annotations tightened across `metrics` and `bbox`
- Dependency version pins relaxed to lower bounds only

## [0.1.0] - 2025-11-25

### Added
- `BBox`: `expand`, `bbox_from_verts` (from IFC vertices), `bbox_list_from_array`, `get_center`, `get_side_planes`, direction vector, project child boxes into parent, `fix_order_points`
- `BBox.points_in_bbox_probability`: probabilistic containment with configurable PDF threshold, returns inlier points and indices
- `metrics.bbox_iou`: bounding box IoU with Munkres-based optimal assignment
- `annotation`: label transfer between point clouds
- `io`: cv4aec format parsing

### Fixed
- Axis alignment bug
- Error handling in `bbox` and `visualization`
- Type checking errors across `bbox` and `visualization`

## [0.0.1] - 2023-07-31

### Added
- Initial installable package structure (`pyproject.toml`, package layout)
- `bbox`: `BBox` class — fitting, manipulation, point containment
- `metrics`: volumetric IoU (`voxel_iou`, `volumetric_iou`), bounding box IoU, centroid deviation
- `visualization`: initial `Visualization` class wrapping open3d
