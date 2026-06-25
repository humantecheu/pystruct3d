# Changelog

All notable changes to pystruct3d are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.17.0] - 2026-06-25

### Changed
- Enabled `ANN` (type-annotation) lint rules for library code; suppressed in `tests/*` and `scripts/*` via per-file-ignores.
- Annotated `BBox.to_cv4aec` parameters (`output_style`, `element_id`, `host_id`) as `str`.
- Annotated deprecated `Visualization.__init__` (`*args: object`, `**kwargs: object`, `-> None`).
- Enabled `PGH003` (blanket `# type: ignore` ban): replaced with `# ty: ignore[unresolved-attribute]` in `visualization.py`; added explicit `import open3d.visualization` so ty resolves the submodule.
- Enabled `INP001`: added `scripts/__init__.py` and `tests/__init__.py`.
- Enabled `FBT001`/`FBT002`: made `progress` keyword-only (`*`) in `io/las.py`, `io/e57.py`, `io/readers.py`, `annotation/utils.py`, and `bbox/bbox.py`.
- Enabled `TC002`/`TC003`: moved `numpy` and `Sequence` imports into `TYPE_CHECKING` block in `visualization.py`.

## [0.16.0] - 2026-06-25

### Added
- SPDX license headers (`# SPDX-License-Identifier: MIT / # Copyright (c) 2023 HumanTech`) added to all Python source files.
- `pre-commit` added as an explicit dev dependency.

### Changed
- Lint config expanded from a selective rule list to `select = ["ALL"]` with an explicit, documented ignore list — broader coverage with every exclusion justified in-place.
- Removed stale ignores `PLR0124`, `PLR0912`, `PLR0915` — confirmed zero violations in the codebase.
- Added `per-file-ignores`: `scripts/*` suppresses T201/PLW0108/C901/EXE001; `tests/*` suppresses S101.
- README Python badge narrowed from `3.10+` to `3.10–3.12` to match `requires-python`.

### Fixed
- Import order corrected in 7 modules (`I001`).
- Unnecessary assignment-before-return inlined in `bbox.py` (`RET504`).
- `voxel_iou.main` and `scripts/visualize_bbox`: replaced legacy `np.random.uniform` with `np.random.default_rng()` (`NPY002`).
- `scripts/evaluate_cv4aec._load`: `open()` replaced with `Path.open()` (`PTH123`).
- `tests/test_bbox_iou`: compound `assert … and …` split into two independent assertions (`PT018`).
- Docs workflow: added `overwrite: true` to `upload-pages-artifact` to prevent stale artifact conflicts.

## [0.15.1] - 2026-06-08

### Fixed
- CI and docs workflows now use `fetch-depth: 0` so setuptools-scm can read tags and report the correct version (was showing `0.1.dev1+g…`).
- CI trigger restricted to `main` branch and PRs only — tag pushes no longer spawn duplicate runs.

### Added
- Docs badge in README linking to GitHub Pages.

## [0.15.0] - 2026-06-08

### Added
- GitHub Actions CI workflow (ruff, ty, pytest on every push/PR).
- GitHub Actions docs workflow: builds pdoc on push to main and deploys to GitHub Pages.
- `pdoc` pre-commit hook: verifies docs build on every commit (no HTML committed).

### Changed
- `BBox.split_bounding_box()` renamed to `BBox.split()` and made a pure function — returns
  `tuple[BBox, BBox]` (first half, second half) without modifying self.
- `BBox.points_in_bbox_probability()`: return type narrowed to
  `tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]`.
- `BBox.from_cv4aec()` / `BBox.to_cv4aec()`: explicit `-> None` / `-> dict` return type added.
- `io.readers._read_open3d_file`: one-line docstring added.

### Fixed
- `testing.generate_bounding_boxes`: replaced `assert n_boxes > 1` with `raise ValueError`.
- `metrics.voxelization_limits.pointcloud_limits`: replaced `assert shape[1] == 3` with `raise ValueError`.

## [0.14.1] - 2026-06-03

### Fixed
- Cap `requires-python` at `<3.13`; open3d does not yet support Python 3.13.

### Changed
- Remove `.python-version` from version control (local dev tool file).

## [0.14.0] - 2026-06-03

### Added
- Logging via Python's standard `logging` module; `NullHandler` registered on the `pystruct3d` root logger — silent by default, opt-in via `logging.basicConfig` or handler configuration
- `progress: bool = True` parameter on `read_las_file`, `read_e57_file`, `read_point_cloud`, and `transfer_labels` — pass `False` to suppress all tqdm output
- `io.las.read_las_file`: rewritten to use `laspy.LasReader.chunk_iterator` for streaming reads; tqdm now shows genuine chunk-level progress on large LAS/LAZ files
- `io.e57.read_e57_file`: tqdm wrapping the scan loop with per-scan point count shown as a postfix; per-scan point count and elapsed time emitted at `DEBUG` level
- INFO-level logging (file name, point count, elapsed time) in `read_las_file`, `read_e57_file`, `_read_open3d_file`, and `transfer_labels`

### Changed
- `BBox.fit_horizontal_aligned`: return type changed from `BBox` to `None` — consistent with all other mutating methods (`rotate`, `translate`, `expand`, …)
- `visualization.Visualizer.show`: merged `show_with_animation` into `show(trajectory=None)` — eliminates duplicated `window_name`/`width`/`height` parameters
- `metrics.voxelization_limits.set_iou` / `weighted_mean_iou`: dropped leading underscore — these functions are imported across module boundaries and are not private

### Removed
- `BBox.iou`: removed to break the geometry → metrics circular dependency; use `pystruct3d.metrics.bbox_iou` directly
- `BBox.fit_minimal`: commented out with a TODO — `NotImplementedError` on a concrete class is a broken contract

### Fixed
- `io.e57.read_e57_file`: replaced `assert` with `raise ValueError` (matches `las.py`)
- `io.cv4aec.bbox_from_cv4aec`: replaced silent `warnings.warn` + implicit zero-BBox return on unknown key with `raise ValueError`

## [0.13.0] - 2026-06-03

### Added
- `testing` module added to README features list
- `tqdm` added as an explicit dependency (was already a transitive dep via open3d)

### Changed
- `BBox.bbox_from_verts` removed — IFC-specific; moved to `openbimxd.geometry.bbox_from_ifc_verts`
- `BBox` methods reorganised into labelled sections (Geometric properties, Containment, Transforms, Fitting, Metrics, Serialisation & I/O, Classmethods, Deprecated)
- `requires-python` lowered from `>=3.12` to `>=3.10` — the true minimum given the `X | Y` union syntax used in annotations
- All dependencies now carry explicit minimum version floors (`open3d>=0.18.0`, `numpy>=1.24.0`, `scipy>=1.11.0`, `laspy>=2.0.0`, `matplotlib>=3.7.0`, `pye57>=0.4.0`, `tqdm>=4.60.0`) to prevent backsliding on older Python versions
- `[tool.setuptools.packages.find]` set to `include = ["pystruct3d*"]` — `tests/` and `scripts/` are now explicitly excluded from the installed wheel
- README: installation now shows `pip install git+https://...`; redundant Requirements section removed
- `BBox.translate_z(**kwargs)` → `BBox.translate_z(*, min_z: float | None = None, max_z: float | None = None)` — explicit typed keyword params; backwards-compatible for all callers using keyword arguments
- `annotation.transfer_labels`: `print()` progress replaced with `tqdm` progress bar
- Completed return-type annotations across `BBox` (`__str__`, `order_points`, `translate`, `translate_z`, `transform_xy`, `expand`, `lower_edges`, `dir_vector_norm`, `axis_align`, `as_np_array`, `get_center_plane`, `get_side_planes`, `length`, `width`, `height`, `fit_axis_aligned`, `get_endpts`)
- Completed parameter-type annotations: `BBox.expand(offset: float)`, `BBox.split_bounding_box(offset: float)`, `BBox.project_into_parent(parent_bbox: BBox)`
- Docstrings added to previously undocumented `BBox` methods: `transform_xy`, `lower_edges`, `dir_vector_norm`, `as_np_array`, `get_center_plane`, `get_side_planes`
- `BBox.length` / `BBox.width`: merged dead second docstring (ASCII art) into the primary docstring as a `::` code block
- `metrics.volumetric_iou.main`: added `-> None` return annotation

### Fixed
- `io.las.read_las_file`: replaced `assert` (disabled by `-O`) with `raise ValueError`
- `io.las`: removed two now-unused `# type: ignore` comments
- `annotation.transfer_labels`: removed bare `print()` calls (violated AGENTS.md "no debug print in library code")

## [0.12.0] - 2026-06-01

### Added
- `bbox/__init__.py`: exports `BBox` — `from pystruct3d.bbox import BBox` now works
- `metrics/__init__.py`: exports `bbox_iou`, `iou_batch`, `match_iou_stats`, `mean_bbox_iou`, `vertex_precision_recall`, `centroid_deviation`
- `annotation/__init__.py`: exports `transfer_labels`
- `preprocessing/__init__.py`: added `downsample` and `density_filter` to re-exports; added `__all__`
- `preprocessing.alignment.align_to_axes`: `seed` parameter (default `0`) to control random state
- `tests/conftest.py`: shared fixtures for the test suite
- `tests/test_preprocessing.py`: 18 tests covering `simple_pca`, `rotate_by_pca`, `crop_roi`, `downsample`, `density_filter`
- `tests/test_bbox.py`: 18 tests covering `BBox` construction, dimensions, fitting, containment, IoU, translate, expand, rotate

### Fixed
- `preprocessing.voxel.downsample`: removed unbounded `np.empty(np.prod(grid_size))` allocation; replaced with `np.searchsorted` — safe for large sparse grids; added empty-input guard
- `preprocessing.voxel.density_filter`: replaced `np.bincount(minlength=np.prod(grid_size))` with `np.unique(return_inverse=True, return_counts=True)` — same fix
- `preprocessing.crop.crop_roi`: replaced fragile negative/positive branch bin-edge logic with uniform `np.floor`; added zero-histogram guard (no longer raises `ZeroDivisionError` on near-empty input)
- `metrics.bbox_iou._pairwise_intersection_2d`: replaced bare `except Exception` with `except (QhullError, ValueError)`
- `bbox.BBox.points_in_bbox`: removed redundant `Exception` from except clause; fixed empty-result index dtype (`np.intp`) to prevent `IndexError`
- `bbox.BBox.expand`: fixed `widht_vector` typo → `width_vector`
- `metrics.point_metric.vertex_precision_recall`: symmetric empty-input guard (handles empty GT as well as empty pred)

### Removed duplications
- `metrics.voxelization_limits`: added private `_set_iou` (set intersection/union) and `_weighted_mean_iou` (GT-weighted average); `mean_voxel_iou` and `mean_volumetric_iou` now delegate to these instead of duplicating the same loop
- `metrics.voxel_iou` and `metrics.volumetric_iou`: both delegate their IoU computation to `_set_iou` — the two functions remain distinct (point cloud voxelization vs BBox interior voxelization)
- `preprocessing.voxel`: extracted private `_to_flat_voxel_indices` helper; `downsample` and `density_filter` both use it instead of duplicating the voxel grid construction
- `bbox.BBox.length` and `bbox.BBox.width`: now delegate to `_dimensions()` instead of each calling `lower_edges()` independently
- `annotation.utils.read_e57_as_point_cloud`: removed — was an unused Open3D wrapper around `io.e57.read_e57_file`; callers can construct an `o3d.PointCloud` directly from the numpy arrays that `read_e57_file` returns. The dead `main()` demo (hardcoded paths) was removed at the same time.
- `bbox/__init__.py`: exports `bbox_list2array` and `bbox_array2list`; `metrics.bbox_iou.mean_bbox_iou` now uses `bbox_list2array` instead of an inline equivalent

### Changed
- `bbox.BBox.points_in_bbox`: removed unused `tolerance` parameter
- `bbox.BBox`: replaced two `# BUG` comments with `# KNOWN LIMITATION:` descriptions
- `io.las`: removed empty `split_pcd_z()` stub
- `preprocessing.labels.filter_ids`: removed redundant `int()` cast on `label_id`
- `preprocessing.labels.labels_to_color`: docstring clarifies modulo wrapping for labels > 20
- `preprocessing.alignment._dominant_wall_angle`: inline comments explain the `< 200` pairs threshold and 10th-percentile fallback

## [0.11.0] - 2026-06-01

### Added
- `preprocessing.pca`: `simple_pca` — numpy eigendecomposition PCA returning mean, eigenvalues, and eigenvectors sorted by descending variance
- `preprocessing.pca`: `rotate_by_pca` — rotate a point cloud so its two largest principal axes align with XY
- `preprocessing.crop`: `crop_roi` — crop a point cloud to its populated XY region via a 2D density histogram

## [0.10.0] - 2026-06-01

### Added
- `metrics.bbox_iou`: `iou_batch` — batch 3D IoU for `(n,8,3)` corner arrays via Sutherland-Hodgman clipping + ConvexHull; ported from `github.com/cv4aec/3d-matching-eval`; 2× faster than the old BBox-loop implementation at n=25
- `metrics.bbox_iou`: `match_iou_stats` — per-structure IoU statistics (min/max/mean/median/std) via optimal IoU assignment
- `metrics.point_metric`: new module for point- and vertex-level metrics
- `metrics.point_metric`: `vertex_precision_recall` — LAP-based precision/recall/F1 at multiple vertex-distance thresholds; fixes an x-component-only distance bug in the original evaluator
- `metrics.point_metric`: `centroid_deviation` — mean nearest-neighbour centroid deviation (moved from `metrics.centroid_deviation`; arg order fixed to GT first)

### Changed
- `metrics.bbox_iou.bbox_iou` and `metrics.bbox_iou.mean_bbox_iou`: delegate to `iou_batch`/`match_iou_stats`; public signatures unchanged; IoU core is now Sutherland-Hodgman + ConvexHull replacing the old line-intersection implementation

### Removed
- `metrics.centroid_deviation`: merged into `metrics.point_metric`

## [0.9.0] - 2026-05-29

### Added
- `preprocessing.voxel`: `downsample` collapses each voxel cell to its centroid and returns a direct O(n) point-to-voxel index mapping (no KDTree)
- `preprocessing.voxel`: `density_filter` removes points whose voxel has fewer than `min_points` occupants

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
