#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
import cv2
import math
import time
import torch
import argparse
import numpy as np

from pathlib import Path
from loguru import logger
from distutils.util import strtobool
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
import ace_vis_util as vutil
from utils.metric import Metric
from ace_network import Regressor
from dataset import CamLocDataset
from ace_visualizer import ACEVisualizer


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'scene',
        type=Path,
        help=
        'path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"'
    )

    parser.add_argument(
        'network',
        type=Path,
        help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path',
                        type=Path,
                        default=Path(__file__).parent /
                        "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session',
                        '-sid',
                        default='',
                        help='custom session name appended to output files, '
                        'useful to separate different runs of a script')

    parser.add_argument('--image_resolution',
                        type=int,
                        default=480,
                        help='base image resolution')

    # ACE is RGB-only, no need for this param.
    # parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2], help='test mode: 1 = RGB, 2 = RGB-D')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument(
        '--hypotheses',
        '-hyps',
        type=int,
        default=64,
        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument(
        '--threshold',
        '-t',
        type=float,
        default=10,
        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument(
        '--inlieralpha',
        '-ia',
        type=float,
        default=100,
        help=
        'alpha parameter of the soft inlier count; controls the softness of the '
        'hypotheses score distribution; lower means softer')

    parser.add_argument(
        '--maxpixelerror',
        '-maxerrr',
        type=float,
        default=100,
        help=
        'maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
        'pose consistency towards all measurements; error is clamped to this value for stability'
    )

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization',
                        type=_strtobool,
                        default=False,
                        help='create a video of the mapping process')

    parser.add_argument(
        '--render_target_path',
        type=Path,
        default='renderings',
        help=
        'target folder for renderings, visualizer will create a subfolder with the map name'
    )

    parser.add_argument(
        '--render_flipped_portrait',
        type=_strtobool,
        default=False,
        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument(
        '--render_sparse_queries',
        type=_strtobool,
        default=False,
        help='set to true if your queries are not a smooth video')

    parser.add_argument(
        '--render_pose_error_threshold',
        type=int,
        default=20,
        help='pose error threshold for the visualisation in cm/deg')

    parser.add_argument(
        '--render_map_depth_filter',
        type=int,
        default=10,
        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument(
        '--render_camera_z_offset',
        type=int,
        default=4,
        help=
        'zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument(
        '--render_frame_skip',
        type=int,
        default=1,
        help='skip every xth frame for long and dense query sequences')

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution,
    )
    logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict,
                                                     head_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name

    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = vutil.get_rendering_target_path(opt.render_target_path,
                                                      opt.network)
        ace_visualizer = ACEVisualizer(
            target_path,
            opt.render_flipped_portrait,
            opt.render_map_depth_filter,
            reloc_vis_error_threshold=opt.render_pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=opt.image_resolution,
        )

        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None
    metric = Metric()
    # Testing loop.
    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33,
                            frame_path) in enumerate(
                                zip(scene_coordinates_B3HW, gt_pose_B44,
                                    intrinsics_B33, filenames)):

                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))

                # Compute the pose via RANSAC.
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose_44[0:3, 3] -
                                         out_pose[0:3, 3]))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                # logger.info(
                #     f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm"
                # )

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err * 100),
                        sparse_query=opt.render_sparse_queries)

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
                pose_log.write(
                    f"{frame_name} "
                    f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                    f"{t[0]} {t[1]} {t[2]} "
                    f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    metric.print()
    pose_log.close()
