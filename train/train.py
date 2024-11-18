# code/train/train.py

import torch
import torch.nn.functional as F
from torch import optim
from tqdm.auto import tqdm
import wandb  # wandb 임포트 추가
from datetime import datetime

from utils.utils import dice_coef, save_model, initialize_wandb
from utils.constants import NUM_EPOCHS, VAL_EVERY, THRESHOLD, CLASSES, WANDB_PROJECT, WANDB_ENTITY
from utils.visualization import visualize_sample, label2rgb
from PIL import Image

def validation(epoch, model, data_loader, criterion, device='cuda'):
    """
    검증 수행 및 Dice 점수 계산.
    """
    print(f'\nStart validation #{epoch:2d}')
    model.to(device)
    model.eval()
    
    tested_images = []
    dices = []
    with torch.no_grad():
        for images, masks, names in tqdm(data_loader, total=len(data_loader)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            h,w = 1363,1800
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != h or output_w != w:
                outputs = F.interpolate(outputs, size=(h, w), mode="bilinear",align_corners=False)
            if mask_h != h or mask_w != w:
                masks = F.interpolate(masks, size=(h, w), mode="bilinear",align_corners=False)
            
            loss = criterion(outputs, masks)
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > THRESHOLD).float()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
    
            show_image = label2rgb(outputs[0].detach().cpu().numpy())
            show_image_name = names[0].split('/')[0]
            tested_images.append(
                wandb.Image(show_image, caption=f'Predicted{show_image_name}'))

    dices = torch.cat(dices, dim=0)
    dices_per_class = dices.mean(dim=0)
    dice_str = "\n".join([f"{c:<12}: {d.item():.4f}" for c, d in zip(CLASSES, dices_per_class)])
    print(dice_str)
    
    # wandb에 Dice 점수 기록
    avg_dice = dices_per_class.mean().item()
    wandb.log({"Validation Average Dice": avg_dice})
    for c, d in zip(CLASSES, dices_per_class):
        wandb.log({f"Validation Dice {c}": d.item(), "Tested Images": tested_images})
    
    return avg_dice

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device='cuda', debug = False):
    """
    세그멘테이션 모델 학습.
    """
    print('Start training...')
    best_dice = 0.0
    use_amp = True

    now = datetime.now()
    timestamp = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
    model.to(device)
    wandb.watch(model)
    scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = []
        torch.cuda.empty_cache()
        for step, (images, masks, names) in enumerate(tqdm(train_loader, total=len(train_loader))):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
            if (step + 1) % 25 == 0:
                current_loss = loss.item()
                print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{step+1}/{len(train_loader)}], Loss: {current_loss:.4f}')
                wandb.log({"Training Loss": current_loss})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch [{epoch}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}')
        wandb.log({"Epoch": epoch, "Average Training Loss": avg_loss})
        
        if epoch % VAL_EVERY == 0:
            avg_dice = validation(epoch, model, valid_loader, criterion, device)
            
            if avg_dice > best_dice:
                print(f'Best Dice updated: {best_dice:.4f} -> {avg_dice:.4f}')
                best_dice = avg_dice
                save_model(model,timestamp,epoch,optimizer,loss,file_name = f'epoch{str(epoch+1)}.pt')
                wandb.log({"Best Dice": best_dice})

