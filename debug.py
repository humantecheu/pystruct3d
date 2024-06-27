import numpy as np
import open3d as o3d

from pystruct3d.visualization import visualization
from pystruct3d.bbox import bbox


def random_testing():
    rand_pts = np.random.uniform(-10, 10, size=(2000000, 3))

    sample_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [5.0, 0.0, 3.0],
            [5.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ]
    )

    door_points = np.array(
        [
            [1.0, -0.2, 1.0],
            [2.0, -0.2, 1.0],
            [2.0, 0.4, 1.0],
            [1.0, 0.4, 1.0],
            [1.0, -0.2, 2.0],
            [2.0, -0.2, 2.0],
            [2.0, 0.4, 2.0],
            [1.0, 0.4, 2.0],
        ]
    )
    visu = visualization.Visualization()
    bx = bbox.BBox(sample_points)
    door_bx = bbox.BBox(door_points)
    door_bx.rotate(-15)
    visu.bbox_geometry([bx, door_bx])
    visu.visualize()
    visu.clear()

    door_bx.project_into_parent(bx)

    visu.bbox_geometry([bx, door_bx])

    visu.visualize()


def compare_bbox_fitting():

    # load point cloud
    pcd = o3d.io.read_point_cloud("/media/kaufmann/scaleBIM/wall_instance.pcd")
    obb = pcd.get_oriented_bounding_box()
    # obb = pcd.get_minimal_oriented_bounding_box()
    obb.color = (0, 0, 1)
    o3d.visualization.draw_geometries([pcd, obb])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # ctr = vis.get_view_control()
    # param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    # opt = vis.get_render_option()

    # vis.add_geometry(pcd)
    # vis.add_geometry(obb)
    # opt.line_width = 5.0
    # opt.background_color = np.asarray([1, 1, 1])
    # ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)  # test
    # vis.run()
    # vis.capture_screen_image("test.jpeg")
    # vis.destroy_window()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 10
    o3d.visualization.draw(
        geometry=[
            {"name": "obb", "geometry": obb, "material": mat},
            {
                "name": "pcd",
                "geometry": pcd,
            },
        ],
        show_skybox=False,
        line_width=5,
    )

    # {
    #     "class_name" : "ViewTrajectory",
    #     "interval" : 29,
    #     "is_loop" : false,
    #     "trajectory" :
    #     [
    #         {
    #             "boundingbox_max" : [ -15.901858125177016, -11.961650463603474, 2.8695255259866066 ],
    #             "boundingbox_min" : [ -16.416921150077872, -19.459678364770983, -0.68938841352936853 ],
    #             "field_of_view" : 60.0,
    #             "front" : [ -0.87437379220136657, 0.43192330712374055, 0.22116222162632715 ],
    #             "lookat" : [ -15.743094961682273, -14.845396682435132, 0.56826789213640849 ],
    #             "up" : [ 0.18453147719163437, -0.12556267551114358, 0.97477287018256542 ],
    #             "zoom" : 0.39999999999999969
    #         }
    #     ],
    #     "version_major" : 1,
    #     "version_minor" : 0
    # }
    hobb = bbox.BBox().fit_horizontal_aligned(np.asarray(pcd.points))
    visu = visualization.Visualization()
    visu.bbox_geometry(hobb, color=[1, 0, 0])
    visu.o3d_point_cloud(pcd)
    visu.visualize()


def main():
    # compare_bbox_fitting()
    random_testing()


if __name__ == "__main__":
    main()
