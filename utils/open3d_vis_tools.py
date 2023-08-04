import argparse
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='open3d')
    parser.add_argument('-pcd1', '--point_cloud_file1', help="point cloud file 1")
    parser.add_argument('-pcd2', '--point_cloud_file2', help="point cloud file 2")
    args = parser.parse_args()
    pcd1 = o3d.io.read_point_cloud(args.point_cloud_file1)
    pcd2 = o3d.io.read_point_cloud(args.point_cloud_file2)
    o3d.visualization.draw_geometries(
        [pcd1, pcd2],
        point_show_normal=True,
        mesh_show_wireframe=True,
        mesh_show_back_face=True,
    )
