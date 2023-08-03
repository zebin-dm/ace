import cv2
import math
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from utils.visualize_localization import Sample, PlotTable

import dsacstar
from ace_visualizer import ACEVisualizer
from pydlutils.basic.yaml import paser_yaml_cfg
from utils.metric import Metric
from registry import ACE_REGISTRY

if __name__ == '__main__':
    device = torch.device("cuda")
    config = paser_yaml_cfg()
    net_cfg = config.net_cfg
    dsac_cfg = ACE_REGISTRY.build(config.dsac_cfg)
    render_cfg = ACE_REGISTRY.build(config.render_cfg)
    exp_cfg = ACE_REGISTRY.build(config.exp_cfg)

    # Setup dataset.
    testset = ACE_REGISTRY.build(config.test_data_cfg)
    logger.info(f'Test images found: {len(testset)}')
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    network = ACE_REGISTRY.build(config.net_cfg)
    network = network.to(device)
    network.eval()

    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = f'{exp_cfg.output_dir}/eval_poses.txt'
    logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")
    pose_log = open(pose_log_file, 'w', 1)

    # Generate video of training process
    ace_visualizer = None
    if render_cfg.visualization:
        ace_visualizer = ACEVisualizer(target_path=f"{exp_cfg.output_dir}/{render_cfg.target_path}",
                                       flipped_portait=render_cfg.flipped_portait,
                                       map_depth_filter=render_cfg.map_depth_filter,
                                       reloc_vis_error_threshold=render_cfg.pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = ACE_REGISTRY.build(config.train_data_cfg)
        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)

        ace_visualizer.setup_reloc_visualisation(frame_count=len(testset),
                                                 data_loader=trainset_loader,
                                                 network=network,
                                                 camera_z_offset=render_cfg.camera_z_offset,
                                                 reloc_frame_skip=render_cfg.frame_skip)

    metric = Metric()
    sample_list = []
    with torch.no_grad():
        for iter_idx, (image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames) in enumerate(testset_loader):
            image_B1HW = image_B1HW.to(device, non_blocking=True)
            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):

                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                # assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])
                fx = intrinsics_33[0, 0]
                fy = intrinsics_33[1, 1]

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))

                # Compute the pose via RANSAC.
                # inlier_count = dsacstar.forward_rgb(
                #     scene_coordinates_3HW.unsqueeze(0),
                #     out_pose,
                #     dsac_cfg.hypotheses,
                #     dsac_cfg.threshold,
                #     focal_length,
                #     ppX,
                #     ppY,
                #     dsac_cfg.inlieralpha,
                #     dsac_cfg.maxpixelerror,
                #     network.OUTPUT_SUBSAMPLE,
                # )
                inlier_count = dsacstar.forward_rgb_v2(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    dsac_cfg.hypotheses,
                    dsac_cfg.threshold,
                    fx,
                    fy,
                    ppX,
                    ppY,
                    dsac_cfg.inlieralpha,
                    dsac_cfg.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )
                # Calculate translation error
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi
                sample_list.append(Sample(imf=frame_path, pose=gt_pose_44, r_err=r_err, t_err=t_err))
                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(query_pose=gt_pose_44.numpy(),
                                                      query_file=frame_path,
                                                      est_pose=out_pose.numpy(),
                                                      est_error=max(r_err, t_err * 100),
                                                      sparse_query=render_cfg.sparse_query)

                metric.update(t_err, r_err)
                # Write estimated pose to pose file (inverse).
                out_pose = out_pose.inverse()

                # Translation.
                t = out_pose[0:3, 3]

                # Rotation to axis angle.
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle

                # Axis angle to quaternion.
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis
                # Write to output file. All in a single line.
                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0]} {q_xyz[1]} {q_xyz[2]} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")
            if iter_idx % 100 == 0:
                logger.info(f"eval progress: {iter_idx}/{len(testset_loader)}")

    metric.print()
    pose_log.close()
    plot_table = PlotTable()
    plot_table.plot_result(sample_list)
