# code/utils/constants.py

import os
import albumentations as A

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
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

#models
ENCODER_NAME = "tu-xception41"
ENCODER_WEIGHT = "imagenet"
IN_CHANNELS = 3

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-4
RANDOM_SEED = 21
NUM_EPOCHS = 30
VAL_EVERY = 2  # Typically validate every epoch
THRESHOLD = 0.5
LOSS_WEIGHT = [0.3,0.4,0.3]

#train
TRAIN_TRANSFORM = A.Compose([
        A.Resize(416*2, 656*2,always_apply=True),
        A.ElasticTransform(),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.3),
        #A.Normalize(),
    ], additional_targets={'mask': 'mask'})

VALID_TRANSFORM = A.Compose([
        A.Resize(416*2, 656*2,always_apply=True),
        #A.Normalize(),
    ])
CROP_HAND = True
RIGHT_HAND = False

# Directories
SAVED_DIR = os.path.join(BASE_DIR, 'code', 'checkpoints')
OUTPUT_CSV = os.path.join(BASE_DIR, 'code', 'output', 'double_scale.csv')
CHECKPOINT = os.path.join('2024-11-18_01-55-18','epoch21.pt')

# Ensure directories exist
os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'code', 'output'), exist_ok=True)

# wandb 설정
WANDB_PROJECT = "Seungsoo semantic-segmentation"
WANDB_ENTITY = "Hanseungsoo63"