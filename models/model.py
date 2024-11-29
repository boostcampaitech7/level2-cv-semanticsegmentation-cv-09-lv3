# code/models/model.py

import torch.nn as nn
from torchvision import models
import os
import torch
import segmentation_models_pytorch as smp
from utils.constants import ENCODER_NAME, ENCODER_WEIGHT, IN_CHANNELS, RESUME

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
    
    print(f"Model Name: {model.__class__.__name__}")

    if resume is not None:
        resume_path = resume
        print(f'load from {resume_path}')
        model.load_state_dict(torch.load(resume_path)['model_state_dict'])
    return model
