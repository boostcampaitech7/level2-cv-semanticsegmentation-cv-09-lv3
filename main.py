# code/main.py

import argparse
import torch
from torch.utils.data import DataLoader
import os
import wandb  # wandb 임포트 추가

from data import XRayDataset, XRayInferenceDataset, get_train_transforms, get_valid_transforms
from models import get_model
from train import train_model
from inference import test_model, save_predictions_to_csv
from utils import set_seed, CLASSES, IMAGE_ROOT_TEST, initialize_wandb, finalize_wandb, loss
from utils.constants import (
    IMAGE_ROOT_TRAIN, LABEL_ROOT, BATCH_SIZE, LR, RANDOM_SEED, NUM_EPOCHS, VAL_EVERY,
    WANDB_PROJECT, WANDB_ENTITY, CHECKPOINT, LABEL_ROOT_TEST, CROP_HAND, RIGHT_HAND, RESUME
)
import segmentation_models_pytorch as smp
from lion_pytorch import Lion

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_png_json_pairs(image_root, label_root):
    """
    PNG 및 JSON 파일 경로를 가져오고 정렬합니다.
    """
    pngs = sorted([
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    ])
    
    jsons = sorted([
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    ])
    
    # 페어링 확인
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    
    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0, "일부 JSON에 대응하는 PNG가 없습니다."
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0, "일부 PNG에 대응하는 JSON이 없습니다."
    
    return pngs, jsons

def main(args):
    """
    학습 또는 추론 모드를 실행하는 메인 함수.
    """
    set_seed(RANDOM_SEED)
    
    if args.mode == 'train':
        # wandb 초기화
        config = {
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": NUM_EPOCHS,
            "val_every": VAL_EVERY,
            "classes": len(CLASSES)
        }
        initialize_wandb(config)
        
        # 데이터 준비
        pngs, jsons = get_png_json_pairs(IMAGE_ROOT_TRAIN, LABEL_ROOT)
        train_transforms = get_train_transforms()
        valid_transforms = get_valid_transforms()
        
        train_dataset = XRayDataset(pngs, jsons, is_train=True, transforms=train_transforms,crop_hand=CROP_HAND)
        valid_dataset = XRayDataset(pngs, jsons, is_train=False, transforms=valid_transforms,crop_hand=CROP_HAND)
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=4,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            dataset=valid_dataset, 
            batch_size=4,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        
        # 모델, 손실 함수, 옵티마이저 초기화
        model = get_model(num_classes=len(CLASSES), pretrained=True, resume=RESUME)
        criterion = loss.Semantic_loss_functions().calc_loss
        optimizer = Lion(params=model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        # 학습
        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler)
        
        # wandb 종료
        finalize_wandb()
        
    elif args.mode == 'inference':
        # wandb 초기화 (옵션: 추론 로그를 기록하고 싶을 때)
        initialize_wandb({
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "mode": "offline"  # 추론 시 wandb 로그를 원하지 않으면 'offline'으로 설정
        })
        
        # 모델 로드
        model = get_model(num_classes=len(CLASSES), pretrained=False, resume=None)
        model_path = os.path.join("code/checkpoints", CHECKPOINT)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        
        # 테스트 데이터 준비
        pngs_test = sorted([
            os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT_TEST)
            for root, _dirs, files in os.walk(IMAGE_ROOT_TEST)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        ])
        
        jsons_test = sorted([
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT_TEST)
        for root, _dirs, files in os.walk(LABEL_ROOT_TEST)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
        ])

        test_transforms = get_valid_transforms()
        test_dataset = XRayInferenceDataset(pngs_test, jsons_test, image_root=IMAGE_ROOT_TEST, label_root = LABEL_ROOT_TEST, transforms=test_transforms, crop_hand=CROP_HAND, right_hand = RIGHT_HAND)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        
        # 추론 - detectoin 쓴 경우 with detection으로 해야함
        #rles, filename_and_class = test_model(model, test_loader)
        rles, filename_and_class = test_model(model, test_loader)
        
        # 결과 CSV 저장
        save_predictions_to_csv(rles, filename_and_class)
        
        # wandb 종료
        finalize_wandb()
        
    else:
        raise ValueError("잘못된 모드입니다. 'train' 또는 'inference'를 선택하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='X-Ray Segmentation Pipeline with wandb')
    parser.add_argument('--mode', type=str, required=True, help="실행 모드: 'train' 또는 'inference'")
    args = parser.parse_args()
    main(args)
