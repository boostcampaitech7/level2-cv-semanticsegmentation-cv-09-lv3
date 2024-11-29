# code/data/dataset.py

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
import albumentations as A

from utils.constants import (
    IMAGE_ROOT_TRAIN, LABEL_ROOT, CLASSES, CLASS2IND
)
from utils.utils import decode_rle_to_mask

class XRayDataset(Dataset):
    def __init__(self, pngs, jsons, is_train=True, transforms=None, crop_hand = False, right_hand=False, n_splits=5, split_num=0):
        """
        Dataset for training and validation.
        
        Parameters:
            pngs (list): List of PNG file paths relative to IMAGE_ROOT_TRAIN.
            jsons (list): List of JSON file paths relative to LABEL_ROOT.
            is_train (bool): If True, creates dataset for training; else for validation.
            transforms (albumentations.Compose): Transformations to apply.
            n_splits (int): Number of splits for GroupKFold.
        """
        self.transforms = transforms
        self.is_train = is_train
        self.crop_hand = crop_hand
        self.right_hand = right_hand

        _filenames = np.array(sorted(pngs))
        _labelnames = np.array(sorted(jsons))
        
        # Split train-valid using GroupKFold
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0] * len(_filenames)  # Dummy labels for GroupKFold
        
        gkf = GroupKFold(n_splits=n_splits)
        train_filenames = []
        train_labelnames = []
        valid_filenames = []
        valid_labelnames = []
        
        for fold, (train_idx, valid_idx) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if fold == split_num:
                    continue

                train_filenames += _filenames[valid_idx].tolist()
                train_labelnames += _labelnames[valid_idx].tolist()

            if fold == split_num and is_train != True:
                valid_filenames = _filenames[valid_idx].tolist() 
                valid_labelnames = _labelnames[valid_idx].tolist()
                break
        
        if is_train:
            self.filenames = train_filenames
            self.labelnames = train_labelnames
        else:
            self.filenames = valid_filenames
            self.labelnames = valid_labelnames
        
        self.images = [None]*len(self.filenames)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT_TRAIN, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
            
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
            
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        x_min,y_min,x_max,y_max = annotations["boxes"]
        annotations = annotations["annotations"]
        
        num_classes = len(CLASSES)
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
            if num_classes>29:
                label[..., -1] += class_label
        inputs = {"image": image, "mask": (label>0).astype(np.uint8)}

        if self.crop_hand:
            crop_transform = A.Crop(x_min,y_min,x_max,2048,always_apply=True)
            inputs = crop_transform(**inputs)
        
        if self.transforms is not None:
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] #if self.is_train else inputs["mask"]
        
        if self.right_hand:
            if item%2 == 1:
                inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
                flip = A.HorizontalFlip(p=1,always_apply=True)
                result = flip(**inputs)

            image = result["image"]
            label = result["mask"]# if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label, image_name

class XRayInferenceDataset(Dataset):
    def __init__(self, pngs, jsons, image_root, label_root, transforms=None, crop_hand=False, right_hand=False):
        """
        Dataset for inference.
        
        Parameters:
            pngs (list): List of PNG file paths relative to image_root.
            image_root (str): Root directory for images.
            transforms (albumentations.Compose): Transformations to apply.
        """
        self.transforms = transforms
        self.image_root = image_root
        self.label_root = label_root
        self.crop_hand = crop_hand
        self.right_hand = right_hand
        self.filenames = sorted(pngs)
        self.labelnames = sorted(jsons)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)

        # label 파일을 읽습니다.

        if self.crop_hand:
            with open(label_path, "r") as f:
                annotations = json.load(f)
            crop_box = annotations["boxes"]
            x_min,y_min,x_max,y_max = crop_box
            image = image[y_min:2048,x_min:x_max]
            inputs = {"image": image}
            crop_box = torch.tensor(crop_box)
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
        
        odd = False
        if self.right_hand:
            odd = item%2==1
            if odd:
                inputs = {"image": image}
                flip = A.HorizontalFlip(p=1,always_apply=True)
                result = flip(**inputs)
                image = result['image']
        flip_ = torch.tensor(odd,dtype = float)

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()

        if self.crop_hand:
            return image, image_name, crop_box, flip_
        else:
            return image, image_name, flip_
        
