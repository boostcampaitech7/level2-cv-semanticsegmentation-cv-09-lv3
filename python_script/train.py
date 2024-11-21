import os
import os.path as osp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import inf
import random
import warnings
import numpy as np
import albumentations as A

import argparse
import torch.optim as optim
# AMP를 위해 torch.cuda.amp 추가
from torch.cuda.amp import GradScaler, autocast

from tqdm.auto import tqdm
from trainer import Trainer
from dataset import XRayDataset
from omegaconf import OmegaConf
from utils.wandb import set_wandb
from torch.utils.data import DataLoader
# AMP를 위해 torch.cuda.amp 추가
from torch.cuda.amp import GradScaler, autocast
from loss.loss_selector import LossSelector
from scheduler.scheduler_selector import SchedulerSelector
from models.model_selector import ModelSelector
from lion_pytorch import Lion

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def setup(cfg):
    # 이미지 파일명
    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.image_root)
        for root, _, files in os.walk(cfg.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    # label json 파일명
    labels = {
        osp.relpath(osp.join(root, fname), start=cfg.label_root)
        for root, _, files in os.walk(cfg.label_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json"
    }

    return np.array(sorted(fnames)), np.array(sorted(labels))


# def get_train_transform():
#     return A.Compose([
#         A.Resize(1536, 1536),
#         A.OneOf([
#             A.RandomBrightnessContrast(p=0.5),
#             A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
#         ], p=0.5),
#         # GaussNoise를 제거하고 다른 변환으로 대체
#         A.OneOf([
#             A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#             A.MotionBlur(blur_limit=3, p=0.5),
#         ], p=0.5),
#         A.OneOf([
#             A.RandomRotate90(p=0.5),
#             A.HorizontalFlip(p=0.5),
#         ], p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
#         A.OneOf([
#             A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
#             A.GridDistortion(p=0.5),
#             A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
#         ], p=0.3),
#         A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
#     ])


def main(cfg):
    set_wandb(cfg)
    set_seed(cfg.seed)

    fnames, labels = setup(cfg)

    transform = [getattr(A, aug)(**params) 
                                         for aug, params in cfg.transform.items()]
    


    train_dataset = XRayDataset(fnames,
                                labels,
                                cfg.image_root,
                                cfg.label_root,
                                fold=cfg.val_fold,
                                transforms=transform,
                                is_train=True)
    
    valid_dataset = XRayDataset(fnames,
                                labels,
                                cfg.image_root,
                                cfg.label_root,
                                fold=cfg.val_fold,
                                transforms=transform,
                                is_train=False)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model 선택
    model_selector = ModelSelector()
    model = model_selector.get_model(cfg.model_name, **cfg.model_parameter)

    model.to(device)

    scaler = GradScaler()

    # optimizer는 고정
    optimizer = optim.Adam(params=model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    
    # scheduler 선택
    scheduler_selector = SchedulerSelector(optimizer)
    scheduler = scheduler_selector.get_scheduler(cfg.scheduler_name, **cfg.scheduler_parameter)
    


    
    # loss 선택
    loss_selector = LossSelector()
    criterion = loss_selector.get_loss(cfg.loss_name, **cfg.loss_parameter)
    
    scaler = GradScaler() if getattr(cfg, 'use_amp', False) else None


    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=valid_loader,
        threshold=cfg.threshold,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_epoch=cfg.max_epoch,
        save_dir=cfg.save_dir,
        val_interval=cfg.val_interval,
        scaler = scaler
    )

    trainer.train()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_train.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    main(cfg)