import os
import numpy as np
from sklearn.model_selection import GroupKFold
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from configs.custom_config.data_vars import get_png, get_json, TRAIN_IMAGE_ROOT, TRAIN_LABEL_ROOT


@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train

        super().__init__(**kwargs)

    def load_data_list(self):
        _filenames = np.array(get_png(is_train=True))
        _labelnames = np.array(get_json())

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break

        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
            data_info = dict(
                img_path=os.path.join(TRAIN_IMAGE_ROOT, img_path),
                seg_map_path=os.path.join(TRAIN_LABEL_ROOT, ann_path),
            )
            data_list.append(data_info)

        return data_list