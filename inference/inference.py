# code/inference/inference.py

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
import os
import wandb  # wandb 임포트 추가
import albumentations as A
import torchvision

from utils.utils import encode_mask_to_rle, decode_rle_to_mask
from utils.constants import IND2CLASS, CLASSES, IMAGE_ROOT_TEST, OUTPUT_CSV, CROP_HAND, RIGHT_HAND

def flip_left_hand(outputs):
    flip = torchvision.transforms.RandomHorizontalFlip(p=1)
    return flip(outputs)

def test_model(model, data_loader, crop_hand = CROP_HAND, right_hand = RIGHT_HAND, device='cuda', threshold=0.5):
    """
    테스트 데이터에 대해 추론 수행.
    """
    model.to(device)
    model.eval()
    
    rles = []
    filename_and_class = []
    tested_images = []
    if crop_hand:
        with torch.no_grad():
            for step, (images, image_names, crop_boxes, flips) in tqdm(enumerate(data_loader), total=len(data_loader)):
                images = images.cuda()    
                outputs = model(images)
                
                outputs = F.interpolate(outputs, size=(1363,1800), mode="bilinear")
                background = F.interpolate(outputs, size=(2048,2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold).float()
                
                raw_image = torch.zeros_like(background)
                for i,(box,output,flip) in enumerate(zip(crop_boxes,outputs, flips)):
                    if flip == torch.tensor(1,dtype = float):
                        output = flip_left_hand(output)
                    bone_mask = output[-1]
                    output[:-1] = output[:-1]*bone_mask

                    x_min, y_min, x_max, y_max = box
                    height = (y_max - y_min).item()
                    width = (x_max - x_min).item()
                    output = output.unsqueeze(0)
                    output = F.interpolate(output,size = (height,width), mode='bilinear')
                    raw_image[i,:,y_min:y_max,x_min:x_max] += output.squeeze(0)

                outputs = raw_image.detach().cpu().numpy()
                outputs = outputs.astype(bool)
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output[:-1]):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name.split('/')[1]}")
    
    else:
        with torch.no_grad():
            for images, image_names, flip in tqdm(data_loader, total=len(data_loader)):
                images = images.to(device)
                outputs = model(images)
                outputs = F.interpolate(outputs, size=(2048, 2048), mode='bilinear', align_corners=False)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold).cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    if flip == torch.tensor(1,dtype = float):
                        output = flip_left_hand(output)
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name.split('/')[1]}")
        
        # wandb에 RLE 결과 기록 (예시로 첫 몇 개만 기록)
        for i in range(min(5, len(rles))):
            wandb.log({f"RLE_{filename_and_class[i]}": rles[i]})
        
    return rles, filename_and_class


def save_predictions_to_csv(rles, filename_and_class, output_csv=OUTPUT_CSV):
    """
    RLE로 인코딩된 예측 결과를 CSV 파일로 저장.
    """
    classes, filenames = zip(*[x.split("_", 1) for x in filename_and_class])
    df = pd.DataFrame({
        "image_name": filenames,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')
    
    # wandb에 CSV 저장 완료 알림 기록
    wandb.log({"CSV Saved": output_csv})
