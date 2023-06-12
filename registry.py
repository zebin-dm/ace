from utils.registry import Registry
from dataset import CamLocDataset
from config import (
    NetConfig,
    DSACStarConfig,
    RenderConfig,
    ExperimentConfig,
)

ACE_REGISTRY = Registry("ACE")
ACE_REGISTRY.register(CamLocDataset)
ACE_REGISTRY.register(NetConfig)
ACE_REGISTRY.register(DSACStarConfig)
ACE_REGISTRY.register(RenderConfig)
ACE_REGISTRY.register(ExperimentConfig)
