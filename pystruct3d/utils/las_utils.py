import laspy
import open3d as o3d
import numpy as np


def read_las_file(
    las_path: str, visualize: int = 0, color_division: float = 65025
) -> o3d.geometry.PointCloud:
    """Read las file

    Args:
        las_path (string): path to las file
        visualize (int): control visualization behavious
        color_division (float): division of color values, 255 or 65025, defaults to 65025

    Returns:
        o3d.geometry.PointCloud: point cloud with points and colors
    """
    assert las_path.endswith(".las") or las_path.endswith(
        ".laz"
    ), "Check the point cloud input type."

    # Load the LAS file
    print(f"Load las / laz point cloud: {las_path}")
    las_file = laspy.read(las_path)

    # Extract the XYZ coordinates and RGB color data
    xyz = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    try:
        rgb = (
            np.vstack((las_file.red, las_file.green, las_file.blue)).transpose()
            / color_division
        )  # 65025
    except AttributeError:
        rgb = np.zeros(xyz.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    if visualize:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            o3d.visualization.draw_geometries(
                [pcd],
                window_name="Input point cloud",
                width=2560,
                height=1440,
            )

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
