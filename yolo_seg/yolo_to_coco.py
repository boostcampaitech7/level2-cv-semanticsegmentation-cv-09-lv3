import yaml
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import os
import shutil
import json


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i+1 for i, v in enumerate(CLASSES)}

def yolo_to_coco(root_path: str, method: str, json_path: str):       
    print("Starting convert yolo to coco format")
    # make folder
    try:
        os.makedirs(f"{root_path}/yolo_train/images")
        os.makedirs(f"{root_path}/yolo_train/labels")
        os.makedirs(f"{root_path}/yolo_valid/images")
        os.makedirs(f"{root_path}/yolo_valid/labels")
    except FileExistsError:
        shutil.rmtree(f"{root_path}/yolo_train")
        shutil.rmtree(f"{root_path}/yolo_valid")
        os.makedirs(f"{root_path}/yolo_train/images")
        os.makedirs(f"{root_path}/yolo_train/labels")
        os.makedirs(f"{root_path}/yolo_valid/images")
        os.makedirs(f"{root_path}/yolo_valid/labels")
        
    # define kfold
    IMAGE_ROOT = os.path.join(root_path, method, "DCM")
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    
    _filenames = np.array(sorted(pngs))
    groups = [os.path.dirname(fname) for fname in _filenames]
    ys = [0 for fname in _filenames]
    gkf = GroupKFold(n_splits=5)
    
    train = []
    valid = []
    
    # split train-valid by GroupKFold
    for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        if i == 0:
            valid += list(_filenames[y])
        else:
            train += list(_filenames[y])
            
    # get json file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    images = json_data["images"]
    annotations = json_data["annotations"]
    
    for _, image in tqdm(enumerate(images)):
        img_id = image["id"]
        width, height = image["width"], image["height"]
        file_name = image["file_name"]
        
        # check train or valid
        check = file_name.split("DCM/")[-1] in train  ## xxx/xxx.png
        folder_name = "yolo_train" if check else "yolo_valid"
        new_file_name = file_name.split("/")[-1]  ## xxx.png
        new_label_name = new_file_name.replace(".png", ".txt")  ###.txt
        
        # copy file
        shutil.copy(
            os.path.join(root_path, method, file_name),
            os.path.join(root_path, folder_name, "images", new_file_name)
        )
        
        # copy json
        candits = [annot for annot in annotations if annot["image_id"] == img_id]
        for candit in candits:
            cls_id = candit["category_id"]
            # bbox = candit["bbox"]
            # bx, by, bw, bh = bbox
            
            # scaled_bx, scaled_by = bx / width, by / height
            # scaled_bw, scaled_bh = bw / width, bh / height
            # scaled_cx, scaled_cy = scaled_bx + scaled_bw / 2, scaled_by + scaled_bh / 2
            # scaled_bbox = [scaled_cx, scaled_cy, scaled_bw, scaled_bh]
            
            seg = candit["segmentation"][0]
            scaled_seg = []
            for i in range(0, len(seg), 2):
                x, y = seg[i], seg[i+1]
                scaled_seg.append(round(x / width, 6))
                scaled_seg.append(round(y / height, 6))
            
            with open(os.path.join(root_path, folder_name, "labels", new_label_name), "a") as f:
                # f.write(f"{cls_id-1} {' '.join(map(str, scaled_bbox))} {' '.join(map(str, scaled_seg))}\n")
                f.write(f"{cls_id-1} {' '.join(map(str, scaled_seg))}\n")
                
    print("Finish convert yolo to coco format")
    print("Train images: ", len(glob(f"{root_path}/yolo_train/images/*")))
    print("Train labels: ", len(glob(f"{root_path}/yolo_train/labels/*")))
    print("Valid images: ", len(glob(f"{root_path}/yolo_valid/images/*")))
    print("Valid labels: ", len(glob(f"{root_path}/yolo_valid/labels/*")))
    

if __name__ == '__main__':
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    root_path = config['DATA_ROOT']
    yolo_to_coco(root_path, "train", f"{root_path}/train_raw.json")