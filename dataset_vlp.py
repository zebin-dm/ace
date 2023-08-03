import os
import random
import cv2
import math
import torch
import numpy as np
import torchvision.transforms.functional as TF

from skimage import io
from skimage import color
from loguru import logger
from torchvision import transforms
from skimage.transform import rotate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class CamLocDatasetVLP(Dataset):
    """Camera localization dataset.
    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(
        self,
        root_dir: str,
        training: bool = False,
        aug_rotation=15,
        aug_scale_min=2 / 3,
        aug_scale_max=3 / 2,
        aug_black_white=0.1,
        aug_color=0.3,
        image_height=480,
        use_half=True,
        num_clusters=None,
        cluster_idx=None,
        debug: bool = False,
        feat_subsample: int = 8,
    ):
        """
        Parameters:
            root_dir: Folder of the data (training or test).
            training: train or test
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees.
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_min: Upper limit of image scale factor for uniform sampling
            aug_black_white: Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]
            aug_color: Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]
            image_height: RGB images are rescaled to this maximum height (if augmentation is disabled, and in the range
                [aug_scale_min * image_height, aug_scale_max * image_height] otherwise).
            use_half: Enabled if training with half-precision floats.
            num_clusters: split the input frames into disjoint clusters using hierarchical clustering in order to train
                an ensemble model. Clustering is deterministic, so multiple training calls with the same number of
                target clusters will result in the same split. See the paper for details of the approach. Disabled by
                default.
            cluster_idx: If num_clusters is not None, then use this parameter to choose the cluster used for training.
            feat_subsample: set defalut for the pretrain feature extractor. do not change, only change with new feature extractor.
        """
        self.feat_subsample = feat_subsample  # the sampe value as ace_network.Regressor.OUTPUT_SUBSAMPLE
        self.use_half = use_half
        self.image_height = image_height
        self.training = training
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color
        self.debug = debug

        self.num_clusters = num_clusters
        self.cluster_idx = cluster_idx
        if self.num_clusters is not None:
            assert self.num_clusters > 0, "num_clusters must be at least 1"
            assert self.cluster_idx is not None, "cluster_idx needs to be specified when num_clusters is set"
            if self.cluster_idx < 0 or self.cluster_idx >= self.num_clusters:
                raise ValueError(f"cluster_idx needs to be between 0 and {self.num_clusters - 1}")

        self.root_dir = root_dir
        self.rgb_dir = f"{root_dir}/rgb"
        self.pose_dir = f"{root_dir}/poses"
        self.calibration_dir = f"{root_dir}/calibration"
        self.file_names = self.get_names(self.pose_dir)

        # We use this to iterate over all frames. If clustering is enabled this is used to filter them.
        self.valid_file_indices = np.arange(len(self.file_names))
        # If clustering is enabled.
        if self.num_clusters is not None:
            logger.info(f"Clustering the {len(self.rgb_files)} into {num_clusters} clusters.")
            _, _, cluster_labels = self._cluster(num_clusters)

            self.valid_file_indices = np.flatnonzero(cluster_labels == cluster_idx)
            logger.info(f"After clustering, chosen cluster: {cluster_idx}, Using {len(self.valid_file_indices)} images.")

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        # statistics calculated over 7scenes training set, should generalize fairly well
        mean = [0.4]
        std = [0.25]
        if self.training:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=self.aug_black_white, contrast=self.aug_black_white),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            # Calculate mean camera center (using the valid frames only).
            self.mean_cam_center = self._compute_mean_camera_center()
            logger.info(f"Load scene: {self.get_scene()} - {self.__len__()}, Mean: {self.mean_cam_center}")
        else:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def get_names(self, pose_path: str):
        name_list = os.listdir(pose_path)
        file_names = []
        for name in name_list:
            if name.endswith("pose.txt"):
                file_names.append(name.replace(".pose.txt", ""))
        file_names.sort()
        logger.info(f"Loading sample: {len(file_names)}")
        return file_names

    def get_rgb(self, idx):
        return f"{self.rgb_dir}/{self.file_names[idx]}.color.png"

    def get_pose(self, idx):
        return f"{self.pose_dir}/{self.file_names[idx]}.pose.txt"

    def get_calibration(self, idx):
        return f"{self.calibration_dir}/{self.file_names[idx]}.calibration.txt"

    @staticmethod
    def _resize_image(image, size):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, size)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _cluster(self, num_clusters):
        """
        Clusters the dataset using hierarchical kMeans.
        Initialization:
            Put all images in one cluster.
        Interate:
            Pick largest cluster.
            Split with kMeans and k=2.
            Input for kMeans is the 3D median scene coordiante per image.
        Terminate:
            When number of target clusters has been reached.
        Returns:
            cam_centers: For each cluster the mean (not median) scene coordinate
            labels: For each image the cluster ID
        """
        num_images = len(self.pose_files)
        logger.info(f'Clustering a dataset with {num_images} frames into {num_clusters} clusters.')

        # A tensor holding all camera centers used for clustering.
        cam_centers = np.zeros((num_images, 3), dtype=np.float32)
        for i in range(num_images):
            pose = self._load_pose(i)
            cam_centers[i] = pose[:3, 3]

        # Setup kMEans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS

        # Label of next cluster.
        label_counter = 0

        # Initialise list of clusters with all images.
        clusters = []
        clusters.append((cam_centers, label_counter, np.zeros(3)))

        # All images belong to cluster 0.
        labels = np.zeros(num_images)

        # iterate kMeans with k=2
        while len(clusters) < num_clusters:
            # Select largest cluster (list is sorted).
            cur_cluster = clusters.pop(0)
            label_counter += 1

            # Split cluster.
            cur_error, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

            # Update cluster list.
            cur_mask = (cur_labels == 0)[:, 0]
            cur_cam_centers0 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

            cur_mask = (cur_labels == 1)[:, 0]
            cur_cam_centers1 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

            cluster_labels = labels[labels == cur_cluster[1]]
            cluster_labels[cur_mask] = label_counter
            labels[labels == cur_cluster[1]] = cluster_labels

            # Sort updated list.
            clusters = sorted(clusters, key=lambda cluster: cluster[0].shape[0], reverse=True)

        # clusters are sorted but cluster indices are random, remap cluster indices to sorted indices
        remapped_labels = np.zeros(num_images)
        remapped_clusters = []

        for cluster_idx_new, cluster in enumerate(clusters):
            cluster_idx_old = cluster[1]
            remapped_labels[labels == cluster_idx_old] = cluster_idx_new
            remapped_clusters.append((cluster[0], cluster_idx_new, cluster[2]))

        labels = remapped_labels
        clusters = remapped_clusters

        cluster_centers = np.zeros((num_clusters, 3))
        cluster_sizes = np.zeros((num_clusters, 1))

        for cluster in clusters:
            # Compute distance of each cam to the center of the cluster.
            cam_num = cluster[0].shape[0]
            cam_data = np.zeros((cam_num, 3))
            cam_count = 0

            # First compute the center of the cluster (mean).
            for i, cam_center in enumerate(cam_centers):
                if labels[i] == cluster[1]:
                    cam_data[cam_count] = cam_center
                    cam_count += 1

            cluster_centers[cluster[1]] = cam_data.mean(0)

            # Compute the distance of each cam from the cluster center. Then average and square.
            cam_dists = np.broadcast_to(cluster_centers[cluster[1]][np.newaxis, :], (cam_num, 3))
            cam_dists = cam_data - cam_dists
            cam_dists = np.linalg.norm(cam_dists, axis=1)
            cam_dists = cam_dists**2

            cluster_sizes[cluster[1]] = cam_dists.mean()

            logger.info("Cluster %i: %.1fm, %.1fm, %.1fm, images: %i, mean squared dist: %f" %
                        (cluster[1], cluster_centers[cluster[1]][0], cluster_centers[cluster[1]][1], cluster_centers[cluster[1]][2], cluster[0].shape[0], cluster_sizes[cluster[1]]))

        logger.info('Clustering done.')

        return cluster_centers, cluster_sizes, labels

    def _compute_mean_camera_center(self):
        mean_cam_center = torch.zeros((3, ))

        for idx in self.valid_file_indices:
            pose = self._load_pose(idx)
            mean_cam_center += pose[0:3, 3]
        mean_cam_center /= len(self)
        return mean_cam_center

    def _load_image(self, idx):
        # return color image: HxWx3
        image = io.imread(self.get_rgb(idx))
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        return image

    def _load_pose(self, idx):
        # return 4x4 matrix.
        pose = np.loadtxt(self.get_pose(idx))
        pose = torch.from_numpy(pose).float()
        return pose

    def _load_calibration(self, idx):
        # return 3x3 matrix.
        intrinsics = np.loadtxt(self.get_calibration(idx))
        intrinsics = torch.from_numpy(intrinsics).float()
        return intrinsics

    def get_scene(self):
        name = os.path.basename(os.path.dirname(self.root_dir))
        return name

    def _get_single_item(self, idx, image_height):
        idx = self.valid_file_indices[idx]
        image = self._load_image(idx)  # HxWx3
        # Load intrinsics.
        intrinsics = self._load_calibration(idx)  # 3x3
        scale_factor = image_height / image.shape[0]
        image_width = int(image.shape[1] * scale_factor)
        new_size = (image_height, image_width)

        if self.debug:
            debug_image = image.copy()
            debug_image = cv2.resize(debug_image, [image_width, image_height])
            logger.info(f"origin imgae shape: {debug_image.shape}, {debug_image.dtype}")

        image = self._resize_image(image, new_size)
        intrinsics_scale = torch.tensor([scale_factor, scale_factor, 1.0]).to(intrinsics)
        intrinsics_scale = torch.diag_embed(intrinsics_scale)
        intrinsics = torch.mm(intrinsics_scale, intrinsics)

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image.size[1], image.size[0]))

        # Apply remaining transforms.
        image = self.image_transform(image)
        # Load pose.
        pose = self._load_pose(idx)

        # Apply data augmentation if necessary.
        if self.training:
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)
            if self.debug:
                debug_image = debug_image / 255.0
                debug_image = rotate(debug_image, angle, order=1, mode='reflect')
                logger.info(f"rotate imgae shape: {debug_image.shape}, {debug_image.dtype}")
                debug_image = debug_image * 255.0
                debug_image = debug_image.astype(np.uint8)

            image = self._rotate_image(image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')
            # Rotate ground truth camera pose as well.
            angle = angle * math.pi / 180.
            # Create a rotation matrix.
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)
            # Apply rotation matrix to the ground truth camera pose.
            pose = torch.matmul(pose, pose_rot)

        if self.use_half:
            image = image.half()
        # Binarize the mask.
        image_mask = image_mask > 0
        pose_inv = pose.inverse()
        intrinsics_inv = intrinsics.inverse()
        coords = 0  # Default for ACE, we don't need them.
        imf = self.get_rgb(idx)
        if self.debug:
            return debug_image, pose, intrinsics, imf
        return image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, imf

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        scale_factor = 1
        if self.training:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)

        image_height = int(self.image_height * scale_factor)
        if type(idx) == list:
            tensors = [self._get_single_item(i, image_height) for i in idx]
            return default_collate(tensors)
        else:
            return self._get_single_item(idx, image_height)


if __name__ == "__main__":
    from utils.visualize import visualize_ep
    dataset = CamLocDatasetVLP(
        root_dir="/mnt/nas/share-all/caizebin/03.dataset/ace/dstpath/20220928T170202+0800_Capture_Xiaomi_21051182C_no2_office_table_full_2",
        feat_subsample=8,
        image_height=1080,
        training=True,
        debug=True,
    )
    imf = []
    pose = []
    intrinsic = []
    image = []
    for didx, data in enumerate(dataset):
        imf.append(data[3])
        pose.append(data[1].numpy())
        intrinsic.append(data[2].numpy())
        image.append(data[0])
        cv2.imwrite(f"debug_image_{didx}.jpg", data[0])

        if didx >= 5:
            break
    print(imf)
    visualize_ep(imf[2], imf[4], pose[2], pose[4], intrinsic[2], intrinsic[4], im1=image[2], im2=image[4])
