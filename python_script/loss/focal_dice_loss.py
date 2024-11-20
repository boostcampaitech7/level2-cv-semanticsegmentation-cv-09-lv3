# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(inputs , targets , alpha=.25, gamma=2) : 
    BCE = F.binary_cross_entropy_with_logits(inputs, targets)
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def focal_dice_loss(pred, target, focal_weight = 0.5):
    focal = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    return loss

class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        focal = focal_loss(pred, target)
        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target)
        loss = focal * self.focal_weight + dice * (1 - self.focal_weight)
        return loss





