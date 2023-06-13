from utils.registry import Registry
from dataset import CamLocDataset
from config import (
    NetConfig,
    DSACStarConfig,
    RenderConfig,
    ExperimentConfig,
)

from ace_loss import ReproLoss

ACE_REGISTRY = Registry("ACE")
ACE_REGISTRY.register(CamLocDataset)
ACE_REGISTRY.register(NetConfig)
ACE_REGISTRY.register(DSACStarConfig)
ACE_REGISTRY.register(RenderConfig)
ACE_REGISTRY.register(ExperimentConfig)

# loss
ACE_REGISTRY.register(ReproLoss)
