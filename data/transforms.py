# code/data/transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.constants import TRAIN_TRANSFORM, VALID_TRANSFORM

def get_train_transforms():
    """
    Define training data transformations.
    """
    return TRAIN_TRANSFORM

def get_valid_transforms():
    """
    Define validation and test data transformations.
    """
    return VALID_TRANSFORM
