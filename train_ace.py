import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from utils.ace_util import get_pixel_grid, to_homogeneous
from ace_network import Regressor
from ace_visualizer import ACEVisualizer
from utils.utils import set_seed, seed_worker
from registry import ACE_REGISTRY
from utils.configuration import paser_yaml_cfg


class TrainerACE:

    def __init__(self, config):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False
        self.config = config
        self.exp_cfg = ACE_REGISTRY.build(self.config.exp_cfg)
        self.use_half = self.config.train_data_cfg.use_half
        self.render_cfg = ACE_REGISTRY.build(config.render_cfg)

        self.device = torch.device('cuda')
        self.workers_number = 12
        # Setup randomness for reproducibility.
        seed = self.exp_cfg.seed
        set_seed(seed)
        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = self.create_generator(seed + 8191)

        self.iteration = 0
        self.dataset = ACE_REGISTRY.build(self.config.train_data_cfg)

        # Create network using the state dict of the pretrained encoder.
        net_cfg = self.config.net_cfg
        logger.info(f"Loading encoder from: {net_cfg.encoder_path}")
        encoder_state_dict = torch.load(net_cfg.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=net_cfg.num_head_blocks,
            use_homogeneous=net_cfg.use_homogeneous,
        )
        self.regressor = self.regressor.to(self.device)

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.exp_cfg.learning_rate)

        # Setup learning rate scheduler.
        steps_per_epoch = self.exp_cfg.training_buffer_size // self.exp_cfg.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.config.scheduler.max_lr,
                                                       epochs=self.exp_cfg.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=self.config.scheduler.cycle_momentum)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # total number of iterations.
        self.iterations = self.exp_cfg.epochs * self.exp_cfg.training_buffer_size // self.exp_cfg.batch_size

        # reprojection loss function.
        self.config.reproject_loss.total_iterations = self.iterations
        self.repro_loss = ACE_REGISTRY.build(self.config.reproject_loss)

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        render_cfg = self.render_cfg
        self.ace_visualizer = None
        if render_cfg.visualization:
            # infer rendering folder from map file name
            target_path = f"{self.exp_cfg.ouput_dir}/{render_cfg.target_path}"
            self.ace_visualizer = ACEVisualizer(target_path=target_path,
                                                flipped_portait=render_cfg.flipped_portait,
                                                map_depth_filter=render_cfg.map_depth_filter,
                                                mapping_vis_error_threshold=render_cfg.mapping_vis_error_th)
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.exp_cfg.visual_steps + 1,
                render_cfg.render_camera_z_offset,
            )

    @staticmethod
    def create_generator(seed: int, device: torch.device = None):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def train(self):
        """
        Main training method.
        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """
        self.create_training_buffer()
        # Train the regression head.
        for epoch in range(self.exp_cfg.epochs):
            self.run_epoch(epoch)

        self.save_model()
        logger.info('Done without errors. ')

        if self.ace_visualizer is not None:
            # Finalize the rendering by animating the fully trained map.
            vis_dataset = ACE_REGISTRY.build(self.config.vis_data_cfg)
            vis_loader = torch.utils.data.DataLoader(vis_dataset, shuffle=False, num_workers=self.workers_number)
            self.ace_visualizer.finalize_mapping(self.regressor, vis_loader)

    def create_training_buffer(self):
        # Generator used to sample random features (runs on the GPU).
        sampling_generator = self.create_generator(self.exp_cfg.seed + 4095, device=self.device)
        batch_generator = self.create_generator(self.exp_cfg.seed + 1023)
        # used to seed individual workers by the dataloader.
        loader_generator = self.create_generator(self.exp_cfg.seed + 511)
        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(self.dataset, generator=batch_generator),
            batch_size=1,
            drop_last=False,
        )

        trainloader = DataLoader(
            dataset=self.dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            pin_memory=True,
            num_workers=self.workers_number,
            persistent_workers=self.workers_number > 0,
            timeout=60 if self.workers_number > 0 else 0,
        )

        logger.info("Starting creation of the training buffer.")
        buffer_size = self.exp_cfg.training_buffer_size
        feat_type = torch.float16 if self.use_half else torch.float32
        self.training_buffer = {
            'features': torch.empty((buffer_size, self.regressor.feature_dim), dtype=feat_type, device=self.device),
            'target_px': torch.empty((buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((buffer_size, 3, 4), dtype=torch.float32, device=self.device),
            'intrinsics': torch.empty((buffer_size, 3, 3), dtype=torch.float32, device=self.device),
            'intrinsics_inv': torch.empty((buffer_size, 3, 3), dtype=torch.float32, device=self.device)
        }

        self.regressor.eval()
        with torch.no_grad():
            buffer_idx = 0
            dataset_passes = 0
            while buffer_idx < buffer_size:
                dataset_passes += 1
                for image, image_mask, _, gt_pose_inv, intrinsics, intrinsics_inv, _, _ in trainloader:
                    image = image.to(self.device, non_blocking=True)  # Bx1xHxW
                    image_mask = image_mask.to(self.device, non_blocking=True)  # Bx1xHxW
                    gt_pose_inv = gt_pose_inv.to(self.device, non_blocking=True)  # Bx4x4
                    intrinsics = intrinsics.to(self.device, non_blocking=True)  # Bx3x3
                    intrinsics_inv = intrinsics_inv.to(self.device, non_blocking=True)  # Bx3x3

                    with autocast(enabled=self.use_half):
                        features = self.regressor.get_features(image)
                    B, C, H, W = features.shape  # Dimensions after the network's downsampling.
                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask = TF.resize(image_mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask = image_mask.bool()
                    if image_mask.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv[:, :3]
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask = image_mask.float()
                    image_mask_N1 = normalize_shape(image_mask)

                    # Over-sample according to image mask.
                    select_num = self.exp_cfg.samples_per_image * B
                    select_num = min(select_num, buffer_size - buffer_idx)
                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(
                        image_mask_N1.view(-1),
                        select_num,
                        replacement=True,  # TODO(zebin): change to False
                        generator=sampling_generator,
                    )

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + select_num
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k][sample_idxs]

                    buffer_idx = buffer_offset
                    if buffer_idx >= buffer_size:
                        break

    def run_epoch(self, epoch):
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True
        self.regressor.train()
        # Shuffle indices.
        training_buffer_size = self.exp_cfg.training_buffer_size
        random_indices = torch.randperm(training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, training_buffer_size, self.exp_cfg.batch_size):
            batch_end = batch_start + self.exp_cfg.batch_size
            if batch_end > training_buffer_size:
                break

            batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][batch_indices].contiguous(),
                self.training_buffer['target_px'][batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][batch_indices].contiguous(),
                self.training_buffer['intrinsics'][batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][batch_indices].contiguous(),
                epoch=epoch,
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, epoch):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size, channels = features_bC.shape
        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.exp_cfg.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.exp_cfg.depth_min)
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)
        # Compute masks used to ignore invalid pixels, Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.exp_cfg.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.exp_cfg.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.exp_cfg.depth_max
        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1
        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.exp_cfg.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

        if self.iteration % self.exp_cfg.visual_steps == 0:
            # Print status.
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            logger.info(f'Iter: {self.iteration:6d} / Epoch {epoch:03d}|{self.exp_cfg.epochs:03d}, '
                        f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%,')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

    def save_model(self):
        # Saving head model with float16
        head_state_dict = self.regressor.heads.half().state_dict()
        save_path = f"{self.config.exp_cfg.output_dir}/{self.dataset.get_scene()}.pt"
        torch.save(head_state_dict, save_path)
        logger.info(f"Saved trained head weights to: {save_path}")


if __name__ == "__main__":
    config = paser_yaml_cfg()
    trainer = TrainerACE(config)
    trainer.train()
