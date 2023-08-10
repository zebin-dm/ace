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
from pydlutils.thirdparty.colmap import read_write_model


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
        fx, fy, cx, cy = intrinsic.tolist()[:4]
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

    def generate_colmap2ace(self, sparse_name="sparse"):
        sparse_path = f"{self.src_path}/{sparse_name}"
        images_path = f"{self.src_path}/images"
        images_bin_file = f"{sparse_path}/images.bin"
        cameras_bin_file = f"{sparse_path}/cameras.bin"
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

            transform = Transform(quat=image.qvec, pos=image.tvec).inverse()
            save_posef = f"{poses_path}/{file_name}.pose.txt"
            np.savetxt(save_posef, transform.matrix)

            intrinsic = self.gen_intrinsic_matrix(cameras[image.camera_id].params)
            save_calif = f"{calibration}/{file_name}.calibration.txt"
            np.savetxt(save_calif, intrinsic)

    def generate_colmap2ace_v2(
        self,
        mapping_name,
        query_name,
        sparse_name="sparse",
    ):
        sparse_path = f"{self.src_path}/{sparse_name}"
        images_path = f"{self.src_path}/images"
        images_bin_file = f"{sparse_path}/images.bin"
        cameras_bin_file = f"{sparse_path}/cameras.bin"
        images = read_write_model.read_images_binary(images_bin_file)
        cameras = read_write_model.read_cameras_binary(cameras_bin_file)
        logger.info(f"the file number is: {len(images)}")
        mapping_path = f"{self.dst_path}/{mapping_name}"
        query_path = f"{self.dst_path}/{query_name}"

        rgb_mapping = f"{mapping_path}/rgb"
        poses_mapping = f"{mapping_path}/poses"
        calibration_mapping = f"{mapping_path}/calibration"

        rgb_query = f"{query_path}/rgb"
        poses_query = f"{query_path}/poses"
        calibration_query = f"{query_path}/calibration"
        os.makedirs(rgb_mapping, exist_ok=True)
        os.makedirs(poses_mapping, exist_ok=True)
        os.makedirs(calibration_mapping, exist_ok=True)
        os.makedirs(rgb_query, exist_ok=True)
        os.makedirs(poses_query, exist_ok=True)
        os.makedirs(calibration_query, exist_ok=True)
        for idx, (image_id, image) in enumerate(images.items()):
            file_name = os.path.splitext(os.path.basename(image.name))[0]
            src_imf = f"{images_path}/{image.name}"
            transform = Transform(quat=image.qvec, pos=image.tvec).inverse()
            intrinsic = self.gen_intrinsic_matrix(cameras[image.camera_id].params)
            if image.name.startswith(mapping_name):
                save_imf = f"{rgb_mapping}/{file_name}.color.png"
                save_posef = f"{poses_mapping}/{file_name}.pose.txt"
                save_calif = f"{calibration_mapping}/{file_name}.calibration.txt"
            elif image.name.startswith(query_name):
                save_imf = f"{rgb_query}/{file_name}.color.png"
                save_posef = f"{poses_query}/{file_name}.pose.txt"
                save_calif = f"{calibration_query}/{file_name}.calibration.txt"
            shutil.copy(src_imf, save_imf)
            np.savetxt(save_posef, transform.matrix)
            np.savetxt(save_calif, intrinsic)

    def generate_colmap2ace_v3(
        self,
        sparse_name="sparse",
        mapping_names: List = None,
        query_names: List = None,
        save_map_name: List = "mapping",
        save_query_name: str = "query",
        check_unkonw_sess: bool = True,
    ):
        sparse_path = f"{self.src_path}/{sparse_name}"
        images_path = f"{self.src_path}/images"
        images_bin_file = f"{sparse_path}/images.bin"
        cameras_bin_file = f"{sparse_path}/cameras.bin"
        if os.path.exists(images_bin_file):
            images = read_write_model.read_images_binary(images_bin_file)
            cameras = read_write_model.read_cameras_binary(cameras_bin_file)
        else:
            images_txt_file = f"{sparse_path}/images.txt"
            cameras_txt_file = f"{sparse_path}/cameras.txt"
            images = read_write_model.read_images_text(images_txt_file)
            cameras = read_write_model.read_cameras_text(cameras_txt_file)
        logger.info(f"the file number is: {len(images)}")
        mapping_path = f"{self.dst_path}/{save_map_name}"
        query_path = f"{self.dst_path}/{save_query_name}"

        rgb_mapping = f"{mapping_path}/rgb"
        poses_mapping = f"{mapping_path}/poses"
        calibration_mapping = f"{mapping_path}/calibration"

        rgb_query = f"{query_path}/rgb"
        poses_query = f"{query_path}/poses"
        calibration_query = f"{query_path}/calibration"
        os.makedirs(rgb_mapping, exist_ok=True)
        os.makedirs(poses_mapping, exist_ok=True)
        os.makedirs(calibration_mapping, exist_ok=True)
        os.makedirs(rgb_query, exist_ok=True)
        os.makedirs(poses_query, exist_ok=True)
        os.makedirs(calibration_query, exist_ok=True)
        for idx, (image_id, image) in enumerate(images.items()):
            sess_name, image_name = image.name.split("/")
            image_name = os.path.splitext(image_name)[0]

            src_imf = f"{images_path}/{image.name}"
            transform = Transform(quat=image.qvec, pos=image.tvec).inverse()
            intrinsic = self.gen_intrinsic_matrix(cameras[image.camera_id].params)
            if sess_name in mapping_names:
                save_imf = f"{rgb_mapping}/{sess_name}_{image_name}.color.png"
                save_posef = f"{poses_mapping}/{sess_name}_{image_name}.pose.txt"
                save_calif = f"{calibration_mapping}/{sess_name}_{image_name}.calibration.txt"
            elif sess_name in query_names:
                save_imf = f"{rgb_query}/{sess_name}_{image_name}.color.png"
                save_posef = f"{poses_query}/{sess_name}_{image_name}.pose.txt"
                save_calif = f"{calibration_query}/{sess_name}_{image_name}.calibration.txt"
            else:
                if check_unkonw_sess:
                    raise Exception(f"Unkonw sess: {sess_name}")
                else:
                    continue
            shutil.copy(src_imf, save_imf)
            np.savetxt(save_posef, transform.matrix)
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
        intrinsic1 = self.load_pose(calif1)
        intrinsic2 = self.load_pose(calif2)

        visualize_ep(imf1, imf2, pose1, pose2, intrinsic1, intrinsic2)

    def get_max_translation(self, path: str, thresh=10):
        max_pose = 0
        max_name = ""
        number = 0
        pose_path = f"{path}/poses"
        name_list = os.listdir(pose_path)
        for name in name_list:
            posef = f"{pose_path}/{name}"
            pose1 = self.load_pose(posef)
            pose1 = Transform(mat=pose1).inverse().matrix
            trans = pose1[:3, 3]
            norm_trans = np.linalg.norm(trans)
            if norm_trans > max_pose:
                max_pose = norm_trans
                max_name = name
            if norm_trans > thresh:
                print(norm_trans)
                number += 1
        print(f"max_norm: {max_pose}, max_name: {max_name}, out number:{number}/{len(name_list)}")

    def copy(
        self,
        srcpath,
        dstpath,
        thresh=10,
    ):

        rgb_src = f"{srcpath}/rgb"
        poses_src = f"{srcpath}/poses"
        calibration_src = f"{srcpath}/calibration"

        rgb_dst = f"{dstpath}/rgb"
        poses_dst = f"{dstpath}/poses"
        calibration_dst = f"{dstpath}/calibration"
        os.makedirs(rgb_dst)
        os.makedirs(poses_dst)
        os.makedirs(calibration_dst)
        name_list = os.listdir(poses_src)
        for name in name_list:
            name = name.replace(".pose.txt", "")
            posef = f"{poses_src}/{name}.pose.txt"
            pose1 = self.load_pose(posef)
            trans = pose1[:3, 3]
            norm_trans = np.linalg.norm(trans)
            if norm_trans > thresh:
                continue
            shutil.copy(posef, poses_dst)
            rgbf = f"{rgb_src}/{name}.color.png"
            shutil.copy(rgbf, rgb_dst)
            calibrationf = f"{calibration_src}/{name}.calibration.txt"
            shutil.copy(calibrationf, calibration_dst)

    def sample(
        self,
        srcpath,
        thresh=10,
    ):

        rgb_src = f"{srcpath}/rgb"
        poses_src = f"{srcpath}/poses"
        calibration_src = f"{srcpath}/calibration"
        map_path = f"{srcpath}_mapping"
        query_path = f"{srcpath}_query"
        rgb_map = f"{map_path}/rgb"
        poses_map = f"{map_path}/poses"
        calibration_map = f"{map_path}/calibration"
        rgb_query = f"{query_path}/rgb"
        poses_query = f"{query_path}/poses"
        calibration_query = f"{query_path}/calibration"

        os.makedirs(rgb_map)
        os.makedirs(poses_map)
        os.makedirs(calibration_map)

        os.makedirs(rgb_query)
        os.makedirs(poses_query)
        os.makedirs(calibration_query)
        name_list = os.listdir(poses_src)
        name_list.sort()
        for idx, name in enumerate(name_list):
            name = name.replace(".pose.txt", "")
            posef = f"{poses_src}/{name}.pose.txt"
            pose1 = self.load_pose(posef)
            trans = pose1[:3, 3]
            norm_trans = np.linalg.norm(trans)
            if norm_trans > thresh:
                continue
            rgbf = f"{rgb_src}/{name}.color.png"
            calibrationf = f"{calibration_src}/{name}.calibration.txt"
            if idx % 2 == 0:
                shutil.copy(posef, poses_map)
                shutil.copy(rgbf, rgb_map)
                shutil.copy(calibrationf, calibration_map)
            else:
                shutil.copy(posef, poses_query)
                shutil.copy(rgbf, rgb_query)
                shutil.copy(calibrationf, calibration_query)


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


def test_vlpcolmap2ace():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/car/src"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst"
    sessions = [
        "20230420104554_colmap",
        "20230706T150716+0800_Capture_OPPO_PEEM00_1",
    ]
    for sess in sessions:
        print(f"Processing session: {sess}")
        config = {
            "src_path": f"{src_path}/{sess}",
            "dst_path": f"{dst_path}/{sess}",
        }
        if os.path.exists(config["dst_path"]):
            continue

        generator = DataSet2Ace(config)
        generator.generate_colmap2ace(sparse_name="sparse/0")


def test_vlpcolmap2ace_v2():
    src_path = "/mnt/nas/share-map/experiment/zhentao/qiyu_ace_prod/20230710115743/colmap"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass"
    mapping_name = "mobili_qiyu_camera0"
    query_name = "mobili_qiyu_camera_query"
    config = {
        "src_path": src_path,
        "dst_path": dst_path,
    }

    generator = DataSet2Ace(config)
    generator.generate_colmap2ace_v2(
        sparse_name="sparse/0",
        mapping_name=mapping_name,
        query_name=query_name,
    )


def test_vlpcolmap2ace_v2_glass_variation_light():
    src_path = "/mnt/gz01/experiment/zhentao/cross_temporal_prod/20230710115743/colmap"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light"
    mapping_name = "mobili_qiyu_camera0"
    query_name = "mobili_qiyu_camera_query"
    config = {
        "src_path": src_path,
        "dst_path": dst_path,
    }

    generator = DataSet2Ace(config)
    generator.generate_colmap2ace_v2(
        sparse_name="sparse/0",
        mapping_name=mapping_name,
        query_name=query_name,
    )


def test_vlpcolmap2ace_v3_19session():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/car/mapping_rescale/merge_test_v2/colmap"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session"
    # mapping_names = [
    #     "mobili_qiyu_camera0",
    #     "mobili_qiyu_camera1",
    #     "mobili_qiyu_camera3",
    #     "mobili_qiyu_camera4",
    #     "mobili_qiyu_camera6",
    #     "mobili_qiyu_camera7",
    #     "mobili_qiyu_camera9",
    #     "mobili_qiyu_camera10",
    #     "mobili_qiyu_camera12",
    #     "mobili_qiyu_camera13",
    #     "mobili_qiyu_camera15",
    #     "mobili_qiyu_camera16",
    #     "mobili_qiyu_camera18",
    # ]
    # query_names = [
    #     "mobili_qiyu_camera2",
    #     "mobili_qiyu_camera5",
    #     "mobili_qiyu_camera8",
    #     "mobili_qiyu_camera11",
    #     "mobili_qiyu_camera14",
    #     "mobili_qiyu_camera17",
    # ]

    mapping_names = [
        "mobili_qiyu_camera0",
        "mobili_qiyu_camera1",
        "mobili_qiyu_camera3",
        "mobili_qiyu_camera4",
        "mobili_qiyu_camera6",
        "mobili_qiyu_camera7",
        "mobili_qiyu_camera8",
        "mobili_qiyu_camera9",
        "mobili_qiyu_camera10",
        "mobili_qiyu_camera11",
        "mobili_qiyu_camera12",
        "mobili_qiyu_camera13",
        "mobili_qiyu_camera14",
        "mobili_qiyu_camera15",
        "mobili_qiyu_camera16",
        "mobili_qiyu_camera17",
        "mobili_qiyu_camera18",
    ]
    query_names = [
        "mobili_qiyu_camera2",
        "mobili_qiyu_camera5",
    ]
    save_map_name = "mapping17_rescale"
    save_query_name = "query2_rescale"
    config = {
        "src_path": src_path,
        "dst_path": dst_path,
    }

    generator = DataSet2Ace(config)
    generator.generate_colmap2ace_v3(
        sparse_name="sparse",
        mapping_names=mapping_names,
        query_names=query_names,
        save_map_name=save_map_name,
        save_query_name=save_query_name,
    )


def test_vlpcolmap2ace_v3_dense_2session():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/car/mappingml001/rescale/dense_2session/colmap"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/car/mappingml001/dataset/dense_2session"

    mapping_names = [
        "mobili_qiyu_camera0",
        "mobili_qiyu_camera1",
        "mobili_qiyu_camera2",
        "mobili_qiyu_camera3",
    ]
    query_names = [
        "mobili_qiyu_camera4",
        "mobili_qiyu_camera5",
        "mobili_qiyu_camera6",
        "mobili_qiyu_camera7",
    ]
    save_map_name = "mapping"
    save_query_name = "query"
    config = {
        "src_path": src_path,
        "dst_path": dst_path,
    }

    generator = DataSet2Ace(config)
    generator.generate_colmap2ace_v3(
        sparse_name="sparse",
        mapping_names=mapping_names,
        query_names=query_names,
        save_map_name=save_map_name,
        save_query_name=save_query_name,
    )


def test_vlpcolmap2ace_v3_19session_single_query():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/car/mapping/merge_test_v2/colmap"
    dst_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session"
    map_name = "mobili_qiyu_camera0"
    query_name = "mobili_qiyu_camera1"
    mapping_names = [
        map_name,
    ]
    query_names = [
        query_name,
    ]
    config = {
        "src_path": src_path,
        "dst_path": dst_path,
    }

    generator = DataSet2Ace(config)
    generator.generate_colmap2ace_v3(
        sparse_name="sparse",
        mapping_names=mapping_names,
        query_names=query_names,
        save_map_name=map_name,
        save_query_name=query_name,
        check_unkonw_sess=False,
    )


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


def test_ace_visulize_vlpcolmap():
    datapath1 = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light/mobili_qiyu_camera_query"
    datapath2 = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light/mobili_qiyu_camera0"
    name1 = "99949008982"
    name2 = "288388143178"
    visualize = ACEVisualize()
    visualize.visualize(datapath1=datapath1, datapath2=datapath2, name1=name1, name2=name2)


def test_get_max_translation():
    sess_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session/mapping17_rescale"
    thresh = 3
    visualize = ACEVisualize()
    visualize.get_max_translation(sess_path, thresh=thresh)


def test_filter_sess():
    src_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/19session/mapping17"
    thresh = 5
    dst_path = f"{src_path}_filter{thresh}"
    os.makedirs(dst_path)

    visualize = ACEVisualize()
    visualize.copy(src_path, dst_path, thresh=thresh)


def test_sample_sess():
    sess_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/20230420104554_colmap_filter"

    visualize = ACEVisualize()
    visualize.sample(sess_path)


def test_sample_ace_images():
    sesses = [
        "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light/mobili_qiyu_camera0",
        "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light/mobili_qiyu_camera_query",
    ]

    map_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light_mix/mapping"
    query_path = "/mnt/nas/share-all/caizebin/03.dataset/car/dst/glass_variation_light_mix/query"
    os.makedirs(map_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)

    rgb_map = f"{map_path}/rgb"
    poses_map = f"{map_path}/poses"
    calibration_map = f"{map_path}/calibration"
    rgb_query = f"{query_path}/rgb"
    poses_query = f"{query_path}/poses"
    calibration_query = f"{query_path}/calibration"

    os.makedirs(rgb_map)
    os.makedirs(poses_map)
    os.makedirs(calibration_map)

    os.makedirs(rgb_query)
    os.makedirs(poses_query)
    os.makedirs(calibration_query)

    for srcpath in sesses:

        rgb_src = f"{srcpath}/rgb"
        poses_src = f"{srcpath}/poses"
        calibration_src = f"{srcpath}/calibration"

        name_list = os.listdir(poses_src)
        name_list.sort()
        for idx, name in enumerate(name_list):
            name = name.replace(".pose.txt", "")
            posef = f"{poses_src}/{name}.pose.txt"
            # pose1 = self.load_pose(posef)
            # trans = pose1[:3, 3]
            # norm_trans = np.linalg.norm(trans)
            # if norm_trans > thresh:
            #     continue
            rgbf = f"{rgb_src}/{name}.color.png"
            calibrationf = f"{calibration_src}/{name}.calibration.txt"
            if idx % 2 == 0:
                shutil.copy(posef, poses_map)
                shutil.copy(rgbf, rgb_map)
                shutil.copy(calibrationf, calibration_map)
            else:
                shutil.copy(posef, poses_query)
                shutil.copy(rgbf, rgb_query)
                shutil.copy(calibrationf, calibration_query)
