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
import thirdparty.colmap.read_write_model as read_write_model


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


class DataSet2Ace():
    DEFAULT_CONFIG = {
        "src_path": None,
        "dst_path": None,
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

    def generate_vlp2ace(self):
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
            pose = transform.matrix  # ACE pose camera2world
            intrinsic = self.gen_intrinsic_matrix(intrinsic)
            np.savetxt(save_posef, pose)
            np.savetxt(save_calif, intrinsic)

    def generate_colmap2ace(self):
        sparse_path = f"{self.src_path}/sparse"
        images_path = f"{self.src_path}/images"
        images_bin_file = f"{sparse_path}/images.bin"
        cameras_bin_file = f"{sparse_path}/cameras.bin"
        # imfs = glob.glob(f"{images_path}/*.jpg")
        images = read_write_model.read_images_binary(images_bin_file)
        cameras = read_write_model.read_cameras_binary(cameras_bin_file)
        logger.info(f"the file number is: {len(images)}")
        rgb_path = f"{self.dst_path}/rgb"
        poses_path = f"{self.dst_path}/poses"
        calibration = f"{self.dst_path}/calibration"
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(poses_path, exist_ok=True)
        os.makedirs(calibration, exist_ok=True)
        for idx, (image_id, image) in enumerate(images.items()):
            file_name = os.path.splitext(os.path.basename(image.name))[0]
            src_imf = f"{images_path}/{image.name}"
            save_imf = f"{rgb_path}/{file_name}.color.png"
            shutil.copy(src_imf, save_imf)

            transform = Transform(quat=image.qvec, pos=image.tvec)
            save_posef = f"{poses_path}/{file_name}.pose.txt"
            np.savetxt(save_posef, transform.matrix)

            intrinsic = self.gen_intrinsic_matrix(cameras[image.camera_id].params)
            save_calif = f"{calibration}/{file_name}.calibration.txt"
            np.savetxt(save_calif, intrinsic)

    def generate_ace2ace(self):
        src_rgb_path = f"{self.src_path}/rgb"
        src_poses_path = f"{self.src_path}/poses"
        src_calibration = f"{self.src_path}/calibration"
        dst_rgb_path = f"{self.dst_path}/rgb"
        dst_poses_path = f"{self.dst_path}/poses"
        dst_calibration = f"{self.dst_path}/calibration"
        os.makedirs(dst_rgb_path, exist_ok=True)
        os.makedirs(dst_poses_path, exist_ok=True)
        os.makedirs(dst_calibration, exist_ok=True)

        images = glob.glob(f"{src_rgb_path}/*.color.png")
        for idx, image_file in enumerate(images):
            file_name = os.path.basename(image_file).replace(".color.png", "")
            src_imf = f"{src_rgb_path}/{file_name}.color.png"
            save_imf = f"{dst_rgb_path}/{file_name}.color.png"
            shutil.copy(src_imf, save_imf)
            src_posef = f"{src_poses_path}/{file_name}.pose.txt"
            dst_posef = f"{dst_poses_path}/{file_name}.pose.txt"
            shutil.copy(src_posef, dst_posef)

            src_calibrationf = f"{src_calibration}/{file_name}.calibration.txt"
            dst_calibrationf = f"{dst_calibration}/{file_name}.calibration.txt"
            focal_length = float(np.loadtxt(src_calibrationf))

            intrinsics = np.eye(3)
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            # Hardcode the principal point to the centre of the image.
            intrinsics[0, 2] = 640 / 2
            intrinsics[1, 2] = 480 / 2
            np.savetxt(dst_calibrationf, intrinsics)


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

    def visualize(self, datapath1, datapath2, name1, name2):
        rgb_path1 = f"{datapath1}/rgb"
        poses_path1 = f"{datapath1}/poses"
        calibration_path1 = f"{datapath1}/calibration"

        rgb_path2 = f"{datapath2}/rgb"
        poses_path2 = f"{datapath2}/poses"
        calibration_path2 = f"{datapath2}/calibration"
        imf1 = f"{rgb_path1}/{name1}.color.png"
        imf2 = f"{rgb_path2}/{name2}.color.png"
        posef1 = f"{poses_path1}/{name1}.pose.txt"
        posef2 = f"{poses_path2}/{name2}.pose.txt"
        calif1 = f"{calibration_path1}/{name1}.calibration.txt"
        calif2 = f"{calibration_path2}/{name2}.calibration.txt"

        pose1 = self.load_pose(posef1)
        pose2 = self.load_pose(posef2)

        pose1 = Transform(mat=pose1).inverse().matrix
        pose2 = Transform(mat=pose2).inverse().matrix
        intrinsic1 = self.load_pose(calif1)
        intrinsic2 = self.load_pose(calif2)

        visualize_ep(imf1, imf2, pose1, pose2, intrinsic1, intrinsic2)


def test_7scene_chess2ace_test():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/7scenes_ace/7scenes_chess"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/7scenes_chess"
    sessions = [
        "test",
        "train",
    ]
    for sess in sessions:
        config = {
            "src_path": f"{src_path}/{sess}",
            "dst_path": f"{dst_path}/{sess}",
        }
        data_ins = DataSet2Ace(config=config)
        data_ins.generate_ace2ace()


def test_vlp2ace_office_and_hall():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/origin"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath"
    sessions = [
        "20220928T170109+0800_Capture_Xiaomi_21051182C_no2_office_table_full",
        "20220928T170202+0800_Capture_Xiaomi_21051182C_no2_office_table_full_2",
        "20221208T164151+0800_Capture_hall_table_full",
        "20221208T164222+0800_Capture_hall_table_2",
    ]

    for sess in sessions:
        print(f"Processing session: {sess}")
        config = {
            "src_path": f"{src_path}/{sess}",
            "dst_path": f"{dst_path}/{sess}",
        }
        data_ins = DataSet2Ace(config=config)
        data_ins.generate_vlp2ace()


def test_vlp2ace_food_and_printer():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/origin"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath"
    sessions = [
        "20230620T100624+0800_Capture_OnePlus_food",
        "20230620T100904+0800_Capture_OnePlus_food_query",
        "20230620T100947+0800_Capture_OnePlus_printer",
        "20230620T101235+0800_Capture_OnePlus_printer_query",
    ]

    for sess in sessions:
        print(f"Processing session: {sess}")
        config = {
            "src_path": f"{src_path}/{sess}",
            "dst_path": f"{dst_path}/{sess}",
        }
        data_ins = DataSet2Ace(config=config)
        data_ins.generate_vlp2ace()


def test_colmap2ace():
    config = {
        "src_path": "/mnt/nas/share-map/common/public_dataset/image_match/temple_nara_japan/dense",
        "dst_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/temple_nara_japan_test"
    }
    generator = DataSet2Ace(config)
    generator.generate_colmap2ace()


def test_ace_visulize():
    datapath1 = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/7scenes_chess/test"
    datapath2 = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/7scenes_chess/test"
    name1 = "seq-05-frame-000998"
    name2 = "seq-05-frame-000102"
    visualize = ACEVisualize()
    visualize.visualize(datapath1, datapath2, name1, name2)


def test_ace_visulize_vlp():
    datapath1 = "/mnt/nas/share-all/caizebin/03.dataset/ace/hall_gt/dst_path/20221208T164151+0800_Capture_hall_table_full"
    datapath2 = "/mnt/nas/share-all/caizebin/03.dataset/ace/hall_gt/dst_path/20221208T164222+0800_Capture_hall_table_2"
    name1 = "BASE@20221208T164151+0800_Capture_hall_table_full@0000000000000186@image@1000000000000001"
    name2 = "BASE@20221208T164222+0800_Capture_hall_table_2@0000000000000165@image@1000000000000001"

    visualize = ACEVisualize()
    visualize.visualize(datapath1=datapath1, datapath2=datapath2, name1=name1, name2=name2)

    datapath1 = "/mnt/nas/share-all/caizebin/03.dataset/ace/office_gt/dst_path/20220928T170109+0800_Capture_Xiaomi_21051182C_no2_office_table_full"
    datapath2 = "/mnt/nas/share-all/caizebin/03.dataset/ace/office_gt/dst_path/20220928T170202+0800_Capture_Xiaomi_21051182C_no2_office_table_full_2"
    name1 = "BASE@20220928T170109+0800_Capture_Xiaomi_21051182C_no2_office_table_full@000000000000025c@image@1000000000000001"
    name2 = "BASE@20220928T170202+0800_Capture_Xiaomi_21051182C_no2_office_table_full_2@00000000000002a1@image@1000000000000001"
    visualize = ACEVisualize()
    visualize.visualize(datapath1=datapath1, datapath2=datapath2, name1=name1, name2=name2)


def test_ace_visulize_food_and_printer():
    datapath1 = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/20230620T100624+0800_Capture_OnePlus_food"
    datapath2 = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/20230620T100904+0800_Capture_OnePlus_food_query"
    name1 = "BASE@20230620T100624+0800_Capture_OnePlus_food@0000000000000927@image@1000000000000001"
    name2 = "BASE@20230620T100904+0800_Capture_OnePlus_food_query@000000000000014e@image@1000000000000001"
    visualize = ACEVisualize()
    visualize.visualize(datapath1=datapath1, datapath2=datapath2, name1=name1, name2=name2)

    datapath1 = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/20230620T100947+0800_Capture_OnePlus_printer"
    datapath2 = "/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/20230620T101235+0800_Capture_OnePlus_printer_query"
    name1 = "BASE@20230620T100947+0800_Capture_OnePlus_printer@0000000000000866@image@1000000000000001"
    name2 = "BASE@20230620T101235+0800_Capture_OnePlus_printer_query@000000000000014b@image@1000000000000001"
    visualize = ACEVisualize()
    visualize.visualize(datapath1=datapath1, datapath2=datapath2, name1=name1, name2=name2)


def test_ace_visulize_colmap():
    datapath = "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/temple_nara_japan/train"
    name1 = "99929260_2648147496"
    name2 = "99949238_8936938996"
    visualize = ACEVisualize()
    visualize.visualize(datapath, name1, name2)
