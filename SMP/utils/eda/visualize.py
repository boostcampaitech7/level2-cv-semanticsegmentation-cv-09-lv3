import streamlit as st
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

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
def annot_to_mask(image,annotations):
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

def update_image_id_from_selectbox():
    st.session_state.image_id = st.session_state.selectbox_choice

def update_image_id_from_slider():
    st.session_state.image_id = st.session_state.slider_choice

def main():
    st.title("Bone Segmentation Visualization")

    # CSV 파일 목록 가져오기
    IMAGE_ROOT = "../data/train/DCM"
    LABEL_ROOT = "../data/train/outputs_json"

    if os.path.exists(IMAGE_ROOT):

        pngs = {
            os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
            for root, _dirs, files in os.walk(IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
        }

        jsons_fn_prefix = sorted([os.path.splitext(fname)[0].split('/')[0] for fname in jsons])
        pngs_fn_prefix = sorted([os.path.splitext(fname)[0].split('/')[0] for fname in pngs])

        set_id = list(set(pngs_fn_prefix))

        assert len(jsons_fn_prefix) - len(pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix) - len(jsons_fn_prefix) == 0

        pngs = sorted(pngs)
        jsons = sorted(jsons)

        pngs = np.array(pngs)
        jsons = np.array(jsons)

        if 'image_index' not in st.session_state:
            st.session_state.image_id = 'ID001'
            st.session_state.left_or_right = 'right'
            input_index = 0

        # 이미지 선택을 위한 UI 구성
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2.5, 1, 1.5])
        with col1:
            st.session_state.right_or_left = st.radio("left or right",('right','left'))
        with col2:
            pass
        with col3:
            st.session_state.image_id = st.select_slider(
                    "Image ID",
                    options=sorted(set_id),
                    value=st.session_state.image_id,
                    key="slider_choice",on_change=update_image_id_from_slider)
        with col4:
            pass
        with col5:
            st.session_state.image_id = st.selectbox("Image ID",sorted(set_id),
                                            key="selectbox_choice",
                                            on_change=update_image_id_from_selectbox)

        image_index = st.session_state.image_id

        if st.session_state.right_or_left == 'right':
            selected_png = pngs[pngs_fn_prefix.index(image_index)]
            selected_json = jsons[jsons_fn_prefix.index(image_index)]
        
        elif st.session_state.right_or_left == 'left':
            selected_png = pngs[pngs_fn_prefix.index(image_index)+1]
            selected_json = jsons[jsons_fn_prefix.index(image_index)+1]

        json_path = os.path.join(LABEL_ROOT,selected_json)
        image_path = os.path.join(LABEL_ROOT,selected_png)

        if selected_png:
            image_path = os.path.join(IMAGE_ROOT,selected_png)
            image = load_image(image_path)

        if selected_json:
            json_path = os.path.join(LABEL_ROOT,selected_json)
            json = load_json(json_path)

            st.header(f"{image_index} {st.session_state.left_or_right} hand")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                original_image = draw_image(image)
                st.image(original_image, caption=f"Image {st.session_state.image_id} Original")
            
            with col2:
                st.subheader("Segment Image")
                segment_image = draw_segment_image(image,json)
                segment_image = draw_image(segment_image)
                st.image(segment_image, caption=f"Image {st.session_state.image_id} Segmentation")


if __name__ == "__main__":
    main()