import numpy as np
from pathlib import Path


def read_e57_file(e57_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read an E57 file, concatenating all scans into a single point cloud.

    Args:
        e57_path: path to a .e57 file.

    Returns:
        Tuple of (xyz, rgb) arrays shaped (N, 3). RGB is normalised to [0, 1].
    """
    import pye57

    if isinstance(e57_path, str):
        e57_path = Path(e57_path)

    if e57_path.suffix != ".e57":
        raise ValueError(f"File format '{e57_path.suffix}' must be '.e57'.")

    e57 = pye57.E57(str(e57_path))
    xyz_parts, rgb_parts = [], []

    for i in range(e57.scan_count):
        data = e57.read_scan(i, colors=True, ignore_missing_fields=True)

        xyz = np.column_stack((
            data["cartesianX"],
            data["cartesianY"],
            data["cartesianZ"],
        ))
        xyz_parts.append(xyz)

        if "colorRed" in data:
            rgb = np.column_stack((
                data["colorRed"],
                data["colorGreen"],
                data["colorBlue"],
            )).astype(np.float64)
            if np.max(rgb) > np.iinfo(np.uint8).max:
                rgb /= np.iinfo(np.uint16).max
            elif np.max(rgb) > 1:
                rgb /= np.iinfo(np.uint8).max
        else:
            rgb = np.zeros(xyz.shape)
        rgb_parts.append(rgb)

    return np.vstack(xyz_parts), np.vstack(rgb_parts)
