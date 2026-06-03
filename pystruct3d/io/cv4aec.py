"""Converters between BBox and the CV4AEC JSON element format."""

import numpy as np

from pystruct3d.bbox.bbox import BBox


def bbox_from_cv4aec(cv4aec_dict: dict) -> BBox:
    """Create a BBox from a CV4AEC element dictionary.

    Supports two styles:
    - ``start_pt``: wall-style with ``start_pt``, ``end_pt``, ``width``, ``height``
    - ``loc``: door/column-style with ``loc``, ``width``, ``depth``, ``height``, ``rotation``

    Args:
        cv4aec_dict: Dictionary of wall, door, or column parameters.

    Returns:
        BBox constructed from the given parameters.
    """
    box = BBox()
    if "start_pt" in cv4aec_dict:
        start_vec = np.asarray(cv4aec_dict["start_pt"])
        end_vec = np.asarray(cv4aec_dict["end_pt"])
        width = cv4aec_dict["width"]
        height = cv4aec_dict["height"]

        center_dir = end_vec - start_vec
        offset_dir = np.cross(center_dir, np.asarray([0, 0, 1]))
        offset_norm = offset_dir / np.linalg.norm(offset_dir)

        box.corner_points[0] = start_vec + 0.5 * width * offset_norm
        box.corner_points[1] = start_vec - 0.5 * width * offset_norm
        box.corner_points[2] = end_vec + 0.5 * width * offset_norm
        box.corner_points[3] = end_vec - 0.5 * width * offset_norm
        box.corner_points[4] = box.corner_points[0].copy()
        box.corner_points[5] = box.corner_points[1].copy()
        box.corner_points[6] = box.corner_points[2].copy()
        box.corner_points[7] = box.corner_points[3].copy()
        box.corner_points[:-4:, 2] += height
        box.order_points()

    elif "loc" in cv4aec_dict:
        loc_vec = np.asarray(cv4aec_dict["loc"])
        bx_width = cv4aec_dict["width"]
        bx_depth = cv4aec_dict["depth"]
        bx_height = cv4aec_dict["height"]
        rotation = cv4aec_dict["rotation"]

        box.corner_points[0] = loc_vec - np.asarray([
            0.5 * bx_width,
            0.5 * bx_depth,
            0.0,
        ])
        box.corner_points[1] = loc_vec + np.asarray([
            0.5 * bx_width,
            -0.5 * bx_depth,
            0.0,
        ])
        box.corner_points[2] = loc_vec + np.asarray([
            0.5 * bx_width,
            0.5 * bx_depth,
            0.0,
        ])
        box.corner_points[3] = loc_vec + np.asarray([
            -0.5 * bx_width,
            0.5 * bx_depth,
            0.0,
        ])
        box.corner_points[4] = box.corner_points[0].copy()
        box.corner_points[5] = box.corner_points[1].copy()
        box.corner_points[6] = box.corner_points[2].copy()
        box.corner_points[7] = box.corner_points[3].copy()
        box.corner_points[:-4:, 2] += bx_height
        box.order_points()
        if round(rotation, 0) != 0:
            box.rotate(rotation)

    else:
        raise ValueError(
            "bbox_from_cv4aec: no valid key found in dict (expected 'start_pt' or 'loc')."
        )

    return box


def bbox_to_cv4aec(
    box: BBox,
    output_style: str = "start_pt",
    element_id: str = "0",
    host_id: str = "0",
) -> dict:
    """Serialize a BBox to a CV4AEC element dictionary.

    Args:
        box: The bounding box to serialize.
        output_style: ``"start_pt"`` for walls, ``"loc"`` for doors and columns.
        element_id: IFC element ID. Defaults to ``"0"``.
        host_id: IFC host element ID. Defaults to ``"0"``.

    Returns:
        Dictionary in the requested CV4AEC format.
    """
    if output_style == "loc":
        return {
            "id": element_id,
            "width": box.length(),
            "depth": box.width(),
            "height": box.height(),
            "loc": list(np.mean(box.corner_points[0:3], axis=0)),
            "rotation": box.angle(),
            "host_id": host_id,
        }

    side_vector = box.corner_points[2] - box.corner_points[1]
    to_base_vec = 0.5 * side_vector
    return {
        "id": element_id,
        "start_pt": list(box.corner_points[0] + to_base_vec),
        "end_pt": list(box.corner_points[1] + to_base_vec),
        "width": box.width(),
        "height": box.height(),
    }
