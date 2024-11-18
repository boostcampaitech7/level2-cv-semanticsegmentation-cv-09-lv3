# code/models/model.py

import torch.nn as nn
from torchvision import models
import os
import torch
import segmentation_models_pytorch as smp
from utils.constants import ENCODER_NAME, ENCODER_WEIGHT, IN_CHANNELS

def get_model(num_classes, pretrained=True, resume=None):
    """
    Initialize the segmentation model.
    
    Parameters:
        num_classes (int): Number of output classes.
        pretrained (bool): If True, use a model pre-trained on COCO.
        
    Returns:
        torch.nn.Module: Modified segmentation model.
    """
    model = smp.DeepLabV3Plus(encoder_name=ENCODER_NAME,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=ENCODER_WEIGHT,     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=IN_CHANNELS,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=num_classes,                      # model output channels (number of classes in your dataset)
            )

    if resume is not None:
        resume_path = os.path.join('./checkpoints',resume)
        pths = os.listdir(resume_path)
        best_epoch = max([int(p.split('.')[0][5:]) for p in pths])
        model.load_state_dict(torch.load(os.path.join(resume_path,f'epoch{str(best_epoch)}.pt')))
    return model