import os

# 데이터 경로를 입력하세요
TRAIN_IMAGE_ROOT = "../data/train/DCM/"
TRAIN_LABEL_ROOT = "../data/train/outputs_json/"
TEST_IMAGE_ROOT = '../data/test/DCM'

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

def get_png(is_train=True):
    image_root = TRAIN_IMAGE_ROOT if is_train else TEST_IMAGE_ROOT
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    return sorted(pngs)

def get_json():
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=TRAIN_LABEL_ROOT)
        for root, _dirs, files in os.walk(TRAIN_LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    return sorted(jsons)