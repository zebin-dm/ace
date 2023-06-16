import os
import shutil
import numpy as np
from loguru import logger
from pose_transform import Transform
import thirdparty.colmap.read_write_model as read_write_model


class Colmap2Ace():
    DEFAULT_CONFIG = {
        "src_path": "/mnt/nas/share-map/common/public_dataset/image_match/temple_nara_japan/dense",
        "dst_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/temple_nara_japan/train"
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


def test_colmap_2_ace():
    config = {
        "src_path": "/mnt/nas/share-map/common/public_dataset/image_match/temple_nara_japan/dense",
        "dst_path": "/mnt/nas/share-all/caizebin/03.dataset/ace/minimum/dst_path/temple_nara_japan/train"
    }
    generator = Colmap2Ace(config)
    generator.generate()
