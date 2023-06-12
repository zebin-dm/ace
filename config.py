import os
from dataclasses import dataclass


@dataclass
class NetConfig:
    encoder_path: str = None  # pre-trained encoder weights
    head_path: str = None  # head network weights path


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
    flipped_portrait: bool = False  # flag for wayspots dataset where images are sideways portrait
    sparse_queries: bool = False  # set to true if your queries are not a smooth video
    pose_error_threshold: int = 20  # pose error threshold for the visualisation in cm/deg
    map_depth_filter: int = 10  # to clean up the ACE point cloud remove points too far away
    camera_z_offset: int = 4  # zoom out of the scene by moving render camera backwards, in meters
    frame_skip: int = 1  # skip every xth frame for long and dense query sequences


@dataclass
class ExperimentConfig:
    ouput_dir: str = None

    def __post_init__(self):
        os.makedirs(self.ouput_dir, exist_ok=True)
