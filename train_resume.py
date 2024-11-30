# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
import yaml
import wandb

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
from model import create_model

# visualization
import matplotlib.pyplot as plt

from dataloader import XRayDataset
from psuedo_label import *
from loss import create_criterion
from optimizer import create_optim
from scheduler import create_sched
from augmentation import create_augmentation


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


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, file_name='checkpoint.pt'):
    output_path = os.path.join(RESULT_DIR, file_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice,
    }, output_path)


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    set_seed()
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
            # B C H W            
            if step == 0:
                table_data = []
                masks = masks[0].cpu().numpy()  # GPU -> CPU로 이동
                preds = outputs[0].cpu().numpy()  # GPU -> CPU로 이동
                for cls_idx in range(n_class):
                    empty_mask = np.zeros((2048, 2048))
                    mask = masks[cls_idx].astype(np.uint8) * 64
                    pred = preds[cls_idx].astype(np.uint8) * 128
                    
                    empty_mask += mask
                    empty_mask += pred
                    table_data.append([IND2CLASS[cls_idx], wandb.Image(empty_mask)])
                table_data.append(["original", wandb.Image(images[0].permute(1, 2, 0).cpu().numpy())])
                wandb.log({f"val/{config['EXP_NAME']}": wandb.Table(columns=["cls_name", "img"], data=table_data)}, step=epoch)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    wandb.log({"val/loss": total_loss / cnt, "val/dice": avg_dice}, step=epoch)
    for c, d in zip(CLASSES, dices_per_class):
        wandb.log({f"val-class/dice_{c}": d.item()}, step=epoch)
    return avg_dice


def train(model, data_loader, val_loader, criterion, optimizer, scheduler, is_plateau, start_epoch=0, best_dice=0.0):
    print(f'Start training from epoch {start_epoch}..')
    set_seed()
    n_class = len(CLASSES)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        for step, (images, masks) in enumerate(data_loader):            
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()            
            
            if (step + 1) % 10 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
        
        wandb.log({"train/loss": loss.item()}, step=epoch)

        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_checkpoint(model, optimizer, scheduler, epoch+1, best_dice)
        
        if scheduler:
            if is_plateau:
                if (epoch + 1) % VAL_EVERY == 0:
                    scheduler.step(dice)
            else:
                scheduler.step()


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    DATA_ROOT = config['DATA_ROOT']
    IMAGE_ROOT = f"{DATA_ROOT}/train/DCM"
    LABEL_ROOT = f"{DATA_ROOT}/train/outputs_json"
    SAVED_DIR = config['SAVED_DIR']
    EXP_NAME = config['EXP_NAME']
    RESULT_DIR = os.path.join(SAVED_DIR, EXP_NAME)

    if not os.path.exists(RESULT_DIR):                                                           
        os.makedirs(RESULT_DIR)

    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    RANDOM_SEED = config['RANDOM_SEED']
    NUM_EPOCHS = config['NUM_EPOCHS']
    VAL_EVERY = config['VAL_EVERY']
    PSEUDOLABEL_FLAG = config['PSEUDO_LABEL']

    # model 정의
    TYPE = config['TYPE']
    MODEL = config['MODEL']
    ENCODER = config['ENCODER']
    RESIZE = config['RESIZE']
    
    clear_test_data_in_train_path(DATA_ROOT)
    if PSEUDOLABEL_FLAG:
        preprocess(DATA_ROOT, config['OUTPUT_CSV_PATH'])

    augmentation_config = config['augmentation']
    augmentation_name = augmentation_config['name']
    augmentation_params = augmentation_config['params'] or {}
    train_tf = create_augmentation(augmentation_name, resize=RESIZE, **augmentation_params)
    valid_tf = create_augmentation('base', resize=RESIZE)

    train_dataset = XRayDataset(
        IMAGE_ROOT, 
        LABEL_ROOT, 
        is_train=True, 
        transforms=train_tf,
        psuedo_flag=PSEUDOLABEL_FLAG,
    )
    valid_dataset = XRayDataset(
        IMAGE_ROOT, 
        LABEL_ROOT, 
        is_train=False, 
        transforms=valid_tf,
    )
    
    if PSEUDOLABEL_FLAG:
        copy_test_data_to_train_path("data")

    print(len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_workers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # model을 정의
    model = create_model(TYPE, MODEL, ENCODER, CLASSES)
    
    # Loss function을 정의합니다.
    loss_config = config['loss']
    loss_name = loss_config['name']
    loss_params = loss_config['params'] or {}

    # Criterion을 정의합니다.    
    criterion = create_criterion(loss_name, **loss_params)

    # Optimizer Config
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    optimizer_params = optimizer_config['params'] or {}

    # Optimizer를 정의합니다.
    optimizer = create_optim(optimizer_name, model, LR, **optimizer_params)

    # Scheduler Config
    scheduler_config = config['scheduler']
    scheduler_name = scheduler_config['name']
    scheduler_params = scheduler_config['params'] or {}

    # Define Scheduler if available
    scheduler = None
    is_plateau = False
    if scheduler_name != "":
        scheduler, is_plateau = create_sched(scheduler_name, optimizer, NUM_EPOCHS, **scheduler_params)

    # 시드를 설정합니다.
    set_seed()

    CAMPER_ID = config['CAMPER_ID']
    wandb.init(project='XRay_Segmentation', entity='ayeong-chonnam-national-university', name=f"{CAMPER_ID}-{EXP_NAME}", config=config)
    
    # 체크포인트 로드
    RESUME = config.get('RESUME', False)
    if RESUME:
        checkpoint_path = os.path.join(RESULT_DIR, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_dice = checkpoint['best_dice']
            print(f"Loaded checkpoint from epoch {start_epoch} with best_dice {best_dice}")
        else:
            print("No checkpoint found at '{}'".format(checkpoint_path))
            start_epoch = 0
            best_dice = 0.0
    else:
        start_epoch = 0
        best_dice = 0.0

    train(model, train_loader, valid_loader, criterion, optimizer, scheduler, is_plateau, start_epoch, best_dice)
