import os
import cv2
import glob
import torch
import shutil
import numpy as np
from typing import List
from loguru import logger
from pose_transform import Transform
from utils.visualize import visualize_ep


def str_to_float_list(data: str) -> List:
    data = data.strip().strip('\n')
    data = data.split(" ")
    data = list(map(float, data))
    return data


def pose_txt_to_transfrom(txt_file: str, return_intrinsic=False):
    """
    txt_file: store pose info
        line 0: quaternions, x, y, z, w
        line 1: transition, x, y, z
        line 3: intrinsic: fx, fy, cx, cy
    """
    with open(txt_file, "r") as fh:
        data = fh.readlines()
        rotation = str_to_float_list(data[0])
        assert len(rotation) == 4
        transition = str_to_float_list(data[1])
        assert len(transition) == 3
        quat = [rotation[3], rotation[0], rotation[1], rotation[2]]
        transform = Transform(quat=quat, pos=transition)
        if not return_intrinsic:
            return transform
        intrinsic = str_to_float_list(data[2])
        intrinsic = np.asarray(intrinsic)
        assert len(intrinsic) == 4
        return transform, intrinsic


class DataSetVlp2Ace():
    DEFAULT_CONFIG = {
        "src_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/origin/20220318T160216+0800_xvnxa_xvnxa002_JMWtest",
        "dst_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/minimum/train"
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.src_path = self.config["src_path"]
        self.dst_path = self.config["dst_path"]
        os.makedirs(self.config["dst_path"], exist_ok=True)

    def gen_intrinsic_matrix(self, intrinsic: np.ndarray):
        """
          fx  0   cx
          0   fy  cy
          0   0   1
        """
        fx, fy, cx, cy = intrinsic.tolist()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def generate(self):
        imfs = glob.glob(f"{self.src_path}/*.jpg")
        logger.info(f"the file number is: {len(imfs)}")
        rgb_path = f"{self.dst_path}/rgb"
        poses_path = f"{self.dst_path}/poses"
        calibration = f"{self.dst_path}/calibration"
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(poses_path, exist_ok=True)
        os.makedirs(calibration, exist_ok=True)
        imfs.sort()
        for idx, imf in enumerate(imfs):
            im_name = os.path.splitext(os.path.basename(imf))[0]
            save_imf = f"{rgb_path}/{im_name}.color.png"
            save_posef = f"{poses_path}/{im_name}.pose.txt"
            save_calif = f"{calibration}/{im_name}.calibration.txt"
            shutil.copy(imf, save_imf)
            metaf = imf.replace(".jpg", ".txt")
            transform, intrinsic = pose_txt_to_transfrom(metaf, return_intrinsic=True)
            pose = transform.inverse().matrix  # convert camera2world to world2camera
            intrinsic = self.gen_intrinsic_matrix(intrinsic)
            np.savetxt(save_posef, pose)
            np.savetxt(save_calif, intrinsic)


class ACEVisualize():

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_pose(posef):
        pose = np.loadtxt(posef).astype(float)
        return pose

    @staticmethod
    def load_intrinsic(calif, imf):
        im = cv2.imread(imf)
        focal_length = float(np.loadtxt(calif))
        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = focal_length
        intrinsics[1, 1] = focal_length
        # Hardcode the principal point to the centre of the image.
        h, w, c = im.shape
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        return intrinsics.numpy()

    def visualize(self):
        name1 = "seq-05-frame-000998"
        name2 = "seq-05-frame-000999"
        rgb_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/7scenes_ace/7scenes_chess/test/rgb"
        poses_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/7scenes_ace/7scenes_chess/test/poses"
        calibration_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/7scenes_ace/7scenes_chess/test/calibration"
        imf1 = f"{rgb_path}/{name1}.color.png"
        imf2 = f"{rgb_path}/{name2}.color.png"
        posef1 = f"{poses_path}/{name1}.pose.txt"
        posef2 = f"{poses_path}/{name2}.pose.txt"
        calif1 = f"{calibration_path}/{name1}.calibration.txt"
        calif2 = f"{calibration_path}/{name2}.calibration.txt"

        pose1 = self.load_pose(posef1)
        pose2 = self.load_pose(posef2)

        intrinsic1 = self.load_intrinsic(calif1, imf1)
        intrinsic2 = self.load_intrinsic(calif2, imf2)

        visualize_ep(imf1, imf2, pose1, pose2, intrinsic1, intrinsic2)

    def visualize_vlp(self):
        name1 = "BASE@20220318T160216+0800_xvnxa_xvnxa002_JMWtest@16dd6afceb8925f1@camera_left@3403b6cfe30f6f85"
        name2 = "BASE@20220318T160216+0800_xvnxa_xvnxa002_JMWtest@16dd6afcd3aa714a@camera_left@3403b6cfe30f7889"
        rgb_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/minimum/train/rgb"
        poses_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/minimum/train/poses"
        calibration_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/minimum/train/calibration"
        imf1 = f"{rgb_path}/{name1}.color.png"
        imf2 = f"{rgb_path}/{name2}.color.png"
        posef1 = f"{poses_path}/{name1}.pose.txt"
        posef2 = f"{poses_path}/{name2}.pose.txt"
        calif1 = f"{calibration_path}/{name1}.calibration.txt"
        calif2 = f"{calibration_path}/{name2}.calibration.txt"

        pose1 = self.load_pose(posef1)
        pose2 = self.load_pose(posef2)

        intrinsic1 = self.load_pose(calif1)
        intrinsic2 = self.load_pose(calif2)

        visualize_ep(imf1, imf2, pose1, pose2, intrinsic1, intrinsic2)

    def visualize_colmap(self):
        base_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/temple_nara_japan/train"
        name1 = "99929260_2648147496"
        name2 = "99949238_8936938996"
        rgb_path = f"{base_path}/rgb"
        poses_path = f"{base_path}/poses"
        calibration_path = f"{base_path}/calibration"
        imf1 = f"{rgb_path}/{name1}.color.png"
        imf2 = f"{rgb_path}/{name2}.color.png"
        posef1 = f"{poses_path}/{name1}.pose.txt"
        posef2 = f"{poses_path}/{name2}.pose.txt"
        calif1 = f"{calibration_path}/{name1}.calibration.txt"
        calif2 = f"{calibration_path}/{name2}.calibration.txt"

        pose1 = self.load_pose(posef1)
        pose2 = self.load_pose(posef2)

        intrinsic1 = self.load_pose(calif1)
        intrinsic2 = self.load_pose(calif2)

        visualize_ep(imf1, imf2, pose1, pose2, intrinsic1, intrinsic2)


def test_vlp2ace():
    config = {
        "src_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/origin/20220719T161313+0800_xvnxa_xvnxa001_mini_test",
        "dst_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/minimum/test"
    }
    data_ins = DataSetVlp2Ace(config=config)
    data_ins.generate()


def test_ace_visulize():
    visualize = ACEVisualize()
    visualize.visualize()


def test_ace_visulize_vlp():
    visualize = ACEVisualize()
    visualize.visualize_vlp()


def test_ace_visulize_colmap():
    visualize = ACEVisualize()
    visualize.visualize_colmap()
