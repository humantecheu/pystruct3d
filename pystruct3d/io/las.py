import laspy
import numpy as np
import open3d as o3d


def read_las_file(
    las_path: str,
    color_division: float = 65025,
) -> o3d.geometry.PointCloud:
    """Read a LAS / LAZ file into an Open3D point cloud.

    Args:
        las_path: path to a .las or .laz file.
        color_division: divisor for RGB values (255 for 8-bit, 65025 for 16-bit).
            Defaults to 65025.

    Returns:
        Open3D PointCloud with XYZ coordinates and normalised RGB colours.
    """
    assert las_path.endswith(".las") or las_path.endswith(".laz"), (
        "Check the point cloud input type."
    )

    las_file = laspy.read(las_path)

    xyz = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    try:
        rgb = (
            np.vstack((las_file.red, las_file.green, las_file.blue)).transpose()
            / color_division
        )
    except AttributeError:
        rgb = np.zeros(xyz.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def write_las_file(las_path, pcd):
    """writes las file from an open3d point cloud

    Args:
        las_path (string): path to las / laz file
        pcd (open3d.geometry.PointCloud): input point cloud
    """

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    # header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    # 2. Create a Las
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.red = colors[:, 0] * 255
    las.red = las.red.astype("int")
    las.green = colors[:, 1] * 255
    las.geen = las.green.astype("int")
    las.blue = colors[:, 2] * 255
    las.blue = las.blue.astype("int")

    las.write(las_path)


def split_pcd_z():
    pass
