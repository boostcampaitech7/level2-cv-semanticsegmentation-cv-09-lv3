# code/utils/constants.py

import os
import albumentations as A
import cv2
import ttach as tta

# Base directory (assuming this script is run from the 'code' directory)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data paths
IMAGE_ROOT_TRAIN = os.path.join(BASE_DIR, '../data', 'train', 'DCM')
LABEL_ROOT = os.path.join(BASE_DIR, '../data', 'train', 'outputs_json')
IMAGE_ROOT_TEST = os.path.join(BASE_DIR, '../data', 'test', 'DCM')
LABEL_ROOT_TEST = os.path.join(BASE_DIR, '../data', 'test', 'outputs_json')

# Classes
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna', 'Bone'
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

#models
ENCODER_NAME = "tu-xception71" #xception41 -> 71
ENCODER_WEIGHT = "imagenet"
IN_CHANNELS = 3

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-4
RANDOM_SEED = 21
NUM_EPOCHS = 80
VAL_EVERY = 10  # Typically validate every epoch
THRESHOLD = 0.5
LOSS_WEIGHT = [0.3,0.4,0.3]
USE_AMP = True

#train
TRAIN_TRANSFORM = A.Compose([
        A.Resize(656*2, 416*2, always_apply=True),
        A.OneOf([
            A.ElasticTransform(p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-30,30),p=0.3,border_mode = cv2.BORDER_CONSTANT),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.5),
    ], additional_targets={'mask': 'mask'})

VALID_TRANSFORM = A.Compose([
        A.Resize(656*2, 416*2, always_apply=True),
    ])
CROP_HAND = True
RIGHT_HAND = False
WEIGHTED_LOSS = False
RESUME = None

# Directories
SAVED_DIR = os.path.join(BASE_DIR, 'code', 'checkpoints')
OUTPUT_CSV = os.path.join(BASE_DIR, 'code', 'output', 'xception71_epoch80_fold2.csv')
CHECKPOINT = os.path.join('2024-11-27_00-57-34_fold_2','epoch70.pt')

TTA_TRANSFORM = None

# Ensure directories exist
os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'code', 'output'), exist_ok=True)

# wandb 설정
WANDB_PROJECT = "Seungsoo semantic-segmentation"
WANDB_ENTITY = "Hanseungsoo63"