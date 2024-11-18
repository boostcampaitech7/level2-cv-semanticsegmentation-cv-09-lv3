import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from utils.constants import LOSS_WEIGHT

class Semantic_loss_functions(object):
    def __init__(self, mode='multilabel'):
        self.JaccardLoss = smp.losses.JaccardLoss(mode= mode)
        self.DiceLoss = smp.losses.DiceLoss(mode= mode)
        self.FocalLoss = smp.losses.FocalLoss(mode=mode)
    
    def calc_loss(self, pred, target, weight = LOSS_WEIGHT):
        bce = self.FocalLoss(pred, target)
        dice = self.DiceLoss(pred, target)
        jaccard = self.JaccardLoss(pred, target)
        loss = bce * weight[0] + dice * weight[1] + jaccard*weight[2]
        return loss