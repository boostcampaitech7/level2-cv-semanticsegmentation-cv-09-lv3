import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch import inf

class DeepLabV3Plus(nn.Module):
    """
    Base Model Unet
    """
    def __init__(self,
                 **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.model = smp.DeepLabV3Plus(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)