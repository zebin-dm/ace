from utils.registry import Registry
from dataset import CamLocDataset

ACE_REGISTRY = Registry("ACE")
ACE_REGISTRY.register(CamLocDataset)
