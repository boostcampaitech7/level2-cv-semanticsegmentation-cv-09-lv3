# code/utils/utils.py

import os
import random
import datetime
import json
import torch
import numpy as np
import wandb

from .constants import SAVED_DIR, CLASS2IND, IND2CLASS, CLASSES

def dice_coef(y_true, y_pred):
    """
    Compute Dice Coefficient.
    y_true and y_pred are expected to be of shape (N, C, H, W)
    """
    y_true_f = y_true.flatten(2).to('cuda')
    y_pred_f = y_pred.flatten(2).to('cuda')
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 1e-4
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, timestamp, epoch, optimizer, loss, file_name='fcn_resnet50_best_model.pt',):
    save_path = os.path.join(SAVED_DIR,timestamp)
    os.makedirs(save_path,exist_ok=True)
    output_path = os.path.join(save_path, file_name)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, output_path)

def set_seed(seed=21):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def encode_mask_to_rle(mask):
    '''
    Encode binary mask to Run-Length Encoding (RLE).
    
    Parameters:
        mask (np.ndarray): Binary mask (1 - mask, 0 - background).
        
    Returns:
        str: RLE encoded string.
    '''
    pixels = mask.flatten(order='C')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    '''
    Decode Run-Length Encoding (RLE) to binary mask.
    
    Parameters:
        rle (str): RLE encoded string.
        height (int): Height of the mask.
        width (int): Width of the mask.
        
    Returns:
        np.ndarray: Decoded binary mask.
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1  # RLE starts at 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def initialize_wandb(config):
    """
    wandb 초기화 함수.
    
    Parameters:
        config (dict): wandb에 기록할 하이퍼파라미터 및 설정.
    """
    wandb.init(
        project=config.get("WANDB_PROJECT", "Seungsoo semantic-segmentation"),
        entity=config.get("WANDB_ENTITY", "ayeong-chonnam-national-university"),
        config=config
    )

def finalize_wandb():
    """
    wandb 종료 함수.
    """
    wandb.finish()