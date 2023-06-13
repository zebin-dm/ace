from torch.optim.lr_scheduler import OneCycleLR
from .registry import Registry

TORCH_REGISTRY = Registry("TORCH")
TORCH_REGISTRY.register(OneCycleLR)