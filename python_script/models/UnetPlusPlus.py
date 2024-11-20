import torch
from torch import inf
import torch.nn as nn
import segmentation_models_pytorch as smp

class UnetPlusPlusModel(nn.Module):
    """
     Unetplusplus
    """
    def __init__(self,
                 **kwargs):
        super(UnetPlusPlusModel, self).__init__()
        self.model = smp.UnetPlusPlus(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)