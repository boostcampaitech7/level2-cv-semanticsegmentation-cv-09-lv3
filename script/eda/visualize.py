import streamlit as st
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import math

# 클래스 이름 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# 클래스별 색상 정의 (RGB 형식)
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

CLASS2IND = {k:i for i,k in enumerate(CLASSES)}
IND2CLASS = {v:k for k, v in CLASS2IND.items()}

# CSV 파일 로드
@st.cache_data
def load_json(label_path):
    with open(label_path,'r') as f:
        annotations = json.load(f)
    return annotations

# 이미지 로드
@st.cache_data
def load_image(image_path):
    return cv2.imread(image_path)

# Mask 그리기
def annot_to_mask(image, annotations):
    annotation = annotations['annotations']
    img_size = image.shape[:-1]
    masks = np.zeros((len(CLASSES),img_size[0],img_size[1]))
    for i in annotation:
        layer = CLASS2IND[i['label']]
        points = np.array(i["points"])
        class_label = np.zeros((img_size[0],img_size[1]), dtype=np.uint8)
        cv2.fillPoly(class_label, [points],1)
        masks[layer] = class_label
    return masks

#Mask를 RGB로 바꾸기
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image

#이미지 그리기
def draw_segment_image(image,json):
    label = annot_to_mask(image,json)
    labeled_img = label2rgb(label)
    return labeled_img

def draw_image(image):
    return Image.fromarray(image)

def get_file_paths(image_root, label_root):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    pngs = sorted(pngs)
    jsons = sorted(jsons)
    
    return np.array(pngs), np.array(jsons)

def display_hand_images(image_root, label_root, pngs, jsons, image_ids, page):
    start_idx = page * 5
    end_idx = min(start_idx + 5, len(image_ids))
    
    for idx in range(start_idx, end_idx):
        image_id = image_ids[idx]
        st.header(f"Subject {image_id}")
        
        col1, col2 = st.columns(2)
        
        # Right hand
        with col1:
            st.subheader("Right Hand")
            selected_png = pngs[idx*2]
            selected_json = jsons[idx*2]
            
            image_path = os.path.join(image_root, selected_png)
            json_path = os.path.join(label_root, selected_json)
            
            image = load_image(image_path)
            json_data = load_json(json_path)
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.image(draw_image(image), caption="Original")
            with subcol2:
                segment_image = draw_segment_image(image, json_data)
                st.image(draw_image(segment_image), caption="Segmentation")
        
        # Left hand
        with col2:
            st.subheader("Left Hand")
            selected_png = pngs[idx*2 + 1]
            selected_json = jsons[idx*2 + 1]
            
            image_path = os.path.join(image_root, selected_png)
            json_path = os.path.join(label_root, selected_json)
            
            image = load_image(image_path)
            json_data = load_json(json_path)
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.image(draw_image(image), caption="Original")
            with subcol2:
                segment_image = draw_segment_image(image, json_data)
                st.image(draw_image(segment_image), caption="Segmentation")

def main():
    st.title("Bone Segmentation Visualization")
    
    # Data paths
    TRAIN_IMAGE_ROOT = "../../data/train/DCM"
    TRAIN_LABEL_ROOT = "../../data/train/outputs_json"
    TEST_IMAGE_ROOT = "../../data/test/DCM"
    
    # Mode selection
    mode = st.radio("Select Mode", ["Training Data", "Test Data"])
    
    if mode == "Training Data":
        if os.path.exists(TRAIN_IMAGE_ROOT):
            pngs, jsons = get_file_paths(TRAIN_IMAGE_ROOT, TRAIN_LABEL_ROOT)
            
            # Get unique image IDs
            image_ids = sorted(set([os.path.splitext(fname)[0].split('/')[0] for fname in pngs]))
            total_pages = math.ceil(len(image_ids) / 5)
            
            # Page navigation
            col1, col2, col3 = st.columns([2, 4, 2])
            with col1:
                if 'page' not in st.session_state:
                    st.session_state.page = 0
                if st.button("Previous Page") and st.session_state.page > 0:
                    st.session_state.page -= 1
            with col2:
                st.write(f"Page {st.session_state.page + 1} of {total_pages}")
            with col3:
                if st.button("Next Page") and st.session_state.page < total_pages - 1:
                    st.session_state.page += 1
            
            display_hand_images(TRAIN_IMAGE_ROOT, TRAIN_LABEL_ROOT, pngs, jsons, image_ids, st.session_state.page)
    
    else:  # Test Data
        if os.path.exists(TEST_IMAGE_ROOT):
            test_pngs = sorted({
                os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
                for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
                for fname in files
                if os.path.splitext(fname)[1].lower() == ".png"
            })
            
            # Test image selection
            selected_image = st.selectbox("Select Test Image", test_pngs)
            
            if selected_image:
                image_path = os.path.join(TEST_IMAGE_ROOT, selected_image)
                image = load_image(image_path)
                
                # Display original test image
                st.subheader("Test Image")
                st.image(draw_image(image), caption="Original Test Image")
                
                # Prediction model selection (placeholder)
                prediction_model = st.selectbox(
                    "Select Prediction Model",
                    ["Model A", "Model B", "Model C"]  # Replace with actual model options
                )
                
                st.info("Note: Prediction visualization will be implemented once prediction models are available.")

if __name__ == "__main__":
    main()