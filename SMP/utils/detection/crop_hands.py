import torch
import json
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"
TEST_IMAGE_ROOT = "../data/test/DCM"
TEST_LABEL_ROOT = "../data/test/outputs_json"

def crop_hands(image_root=IMAGE_ROOT, json_root = LABEL_ROOT):

    if os.path.exists(image_root):

        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        pngs = sorted(pngs)
        pngs = np.array(pngs)

        jsons = {
            os.path.relpath(os.path.join(root, fname), start=json_root)
            for root, _dirs, files in os.walk(json_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }     

        jsons = sorted(jsons)
        jsons = np.array(jsons)
        assert len(pngs) == len(jsons)

    else:
        print("Wrong Path!!")

    for n,(i,j) in enumerate(zip(pngs,jsons)):
        label_path = os.path.join(json_root, j)
        with open(label_path, "r") as f:
            annotations = json.load(f)

        path = os.path.join(image_root,i)
        img = cv2.imread(path)
        text = 'a hand.'

        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[img.shape[:-1]]
        )

        x_min,y_min,x_max,y_max = [int(i) for i in np.array(results[0]['boxes'][0].cpu())]
        annotations['boxes']=[x_min,y_min,x_max,y_max]

        with open(label_path, "w") as f:
            json.dump(annotations,f,indent=2)

        if n%100 == 0:
            print(f'{str(n)}번째 완료!')


if __name__ == "__main__":
    crop_hands(IMAGE_ROOT,LABEL_ROOT)

