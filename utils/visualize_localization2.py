import cv2
import argparse
import numpy as np
import open3d as o3d


class VisualizeLocalization():

    def __init__(self, gt_path: str, pred_path: str) -> None:

        self.gt_path = gt_path
        self.pred_path = pred_path
        self.mini_t = None
        self.maxi_t = None
        self.mini_r = None
        self.maxi_r = None

        self.gt_meta, self.gt_points, self.gt_normals = self.load_data(self.gt_path)
        self.pred_meta, self.pred_points, self.pred_nomals = self.load_data(self.pred_path)

    def update_range(self, r: float, t: float):
        if self.mini_t is None:
            self.mini_t = t
            self.maxi_t = t
            self.mini_r = r
            self.maxi_r = r
        else:
            self.mini_t = min(self.mini_t, t)
            self.maxi_t = max(self.maxi_t, t)
            self.mini_r = min(self.mini_r, r)
            self.maxi_r = max(self.maxi_r, r)

    def load_data(self, data_path: str):
        meta_file = f"{data_path}/meta.txt"
        with open(meta_file, "r") as fh:
            data = fh.readlines()
        new_data = []
        points = []
        normals = []
        for item in data:

            item = item.strip().strip("\n").split()
            item = (item[0], float(item[1]), float(item[2]))
            new_data.append(item)
            self.update_range(item[1], item[2])

            pose = np.loadtxt(item[0])
            rot, _ = cv2.Rodrigues(pose[0:3, 0:3])
            angle = np.linalg.norm(rot)
            axis = rot / angle
            translation = pose[:3, 3]
            points.append(translation)
            normals.append(axis)
        points = np.asarray(points)
        normals = np.asarray(normals)
        normals = normals.squeeze(axis=2)

        return new_data, points, normals

    @staticmethod
    def create_points(points: np.ndarray, normals: np.ndarray, color: np.ndarray):
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(points)
        points_cloud.normals = o3d.utility.Vector3dVector(normals)
        points_cloud.paint_uniform_color(color)
        return points_cloud

    def visualize(self):
        points_gt = self.create_points(self.gt_points, self.gt_normals, [0, 1, 0])
        points_pre = self.create_points(self.pred_points, self.pred_nomals, [0, 0, 1])
        o3d.io.write_point_cloud(f"{self.gt_path}.pcd", points_gt)
        o3d.io.write_point_cloud(f"{self.pred_path}.pcd", points_pre)
        o3d.visualization.draw_geometries(
            [points_gt, points_pre],
            point_show_normal=True,
            mesh_show_wireframe=True,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    data_gt_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session/query2_rescale_visualize_gt"
    data_pred_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session/query2_rescale_visualize_pred"
    save_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session/query2_rescale_visualize.pcd"
    visualizer = VisualizeLocalization(gt_path=data_gt_path, pred_path=data_pred_path)
    visualizer.visualize()
