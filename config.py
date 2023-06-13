import os
from dataclasses import dataclass


@dataclass
class NetConfig:
    encoder_path: str = None  # pre-trained encoder weights
    head_path: str = None  # head network weights path
    num_head_blocks: int = 1
    use_homogeneous: bool = True


@dataclass
class DSACStarConfig:
    hypotheses: int = 64  # number of hypotheses, i.e. number of RANSAC iterations
    threshold: float = 10  # inlier threshold in pixels (RGB) or centimeters (RGB-D)
    # alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer
    inlieralpha: float = 100
    # maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking
    # pose consistency towards all measurements; error is clamped to this value for stability
    maxpixelerror: float = 100


@dataclass
class RenderConfig:
    visualization: bool = False  # create a video of the mapping process
    target_path: str = "renderings"  # target folder for renderings, visualizer will create a subfolder with the map name
    flipped_portait: bool = False  # flag for wayspots dataset where images are sideways portrait
    sparse_query: bool = False  # set to true if your queries are not a smooth video
    pose_error_threshold: int = 20  # pose error threshold for the visualisation in cm/deg
    mapping_vis_error_th: int = 10  # reprojection error threshold for the visualisation in px
    map_depth_filter: int = 10  # to clean up the ACE point cloud remove points too far away
    camera_z_offset: int = 4  # zoom out of the scene by moving render camera backwards, in meters
    frame_skip: int = 1  # skip every xth frame for long and dense query sequences


@dataclass
class ExperimentConfig:
    output_dir: str = None
    seed: int = 2089
    learning_rate: float = None
    epochs: int = None
    batch_size: int = None

    # For ACE
    training_buffer_size: int = None  # number of patches in the training buffer
    samples_per_image: int = 1024
    visual_steps: int = None  # print loss every n iterations, and (optionally) write a visualisation frame
    depth_min: float = 0.1
    depth_max: float = 1000.0
    depth_target: float = 10.0  # default depth to regularize training
    repro_loss_hard_clamp: float = 1000.0  # hard clamping threshold for the reprojection losses

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
