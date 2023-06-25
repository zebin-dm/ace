from pydlutils.basic.registry import Registry
from dataset import CamLocDataset
from dataset_vlp import CamLocDatasetVLP
from config import (
    DSACStarConfig,
    RenderConfig,
    ExperimentConfig,
)
from ace_network import Regressor
from ace_loss import ReproLoss

ACE_REGISTRY = Registry("ACE")
ACE_REGISTRY.register(CamLocDataset)
ACE_REGISTRY.register(CamLocDatasetVLP)
ACE_REGISTRY.register(DSACStarConfig)
ACE_REGISTRY.register(RenderConfig)
ACE_REGISTRY.register(ExperimentConfig)
ACE_REGISTRY.register(Regressor)

# loss
ACE_REGISTRY.register(ReproLoss)
