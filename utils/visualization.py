# code/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt

from .constants import CLASSES

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb(label):
    """
    Convert label mask to RGB image using predefined palette.
    
    Parameters:
        label (np.ndarray): Label mask of shape (C, H, W).
        
    Returns:
        np.ndarray: RGB image.
    """
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

def visualize_sample(image, label=None):
    """
    Visualize image and its corresponding label mask.
    
    Parameters:
        image (torch.Tensor or np.ndarray): Image tensor or array with shape (C, H, W).
        label (torch.Tensor or np.ndarray, optional): Label mask with shape (C, H, W).
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
        
    # Convert CHW to HWC
    image = np.transpose(image, (1, 2, 0))
    
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    
    if label is not None:
        ax[1].imshow(label2rgb(label))
        ax[1].set_title('Label')
    
    plt.show()
