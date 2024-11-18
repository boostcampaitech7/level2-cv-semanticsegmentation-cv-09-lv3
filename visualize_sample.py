from code.data import XRayDataset, get_train_transforms
from code.utils import visualize_sample, get_png_json_pairs
import os

# Define base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_ROOT_TRAIN = os.path.join(BASE_DIR, 'data', 'train', 'DCM')
LABEL_ROOT = os.path.join(BASE_DIR, 'data', 'train', 'outputs_json')

# Get paired filenames
pngs, jsons = get_png_json_pairs(IMAGE_ROOT_TRAIN, LABEL_ROOT)

# Initialize dataset
dataset = XRayDataset(pngs, jsons, is_train=True, transforms=get_train_transforms())

# Get a sample
image, label = dataset[0]

# Visualize
visualize_sample(image, label)
