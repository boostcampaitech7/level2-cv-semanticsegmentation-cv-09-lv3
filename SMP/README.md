## Installation
PyPI version:
```
$ pip install segmentation-models-pytorch
```
Latest version from source:
```
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

## configs
- `config_resume.YAML`: SMP 모듈을 활용하기 위한 config 파일입니다. 하이퍼파라미터, 모델, Backbone 등을 설정할 수 있습니다.

## datasets
- `augmentation.py`: 다양한 augmentation을 적용하기 위한 코드입니다.
- `convert_to_coco.py`: Dataset을 coco format으로 변경하기 위한 코드입니다.
- `dataloader.py`: load에 필요한 dataset 코드입니다. 다양한 pre, post processing이 가능합니다.

## src
- train, inference 등의 실행파일들이 있는 폴더입니다.
- `train.py`, `train_amp`, `train_resume`: 상황에 맞게 train하기 위한 코드입니다.
- `inference.py`, `inference_origin`, `inference_tta`: inference를 위한 코드입니다.
- `model.py` SMP의 모델을 불러오고 저장하는 코드입니다.

## Utils
- dataset과 model외에 하이퍼파라미터나, Data Processing을 위한 방법론들이 적용됩니다.
- `detection`: data pre-processing 과정에서 Grounding DINO를 활용한 Detection을 구현하는 코드입니다.
- `eda`: 다양한 EDA를 진행하였습니다. 특히 `visualize.py`를 통해서 Streamlit을 활용한 시각화 페이지를 구현하였습니다.
- `loss.py`: SMP 모듈 학습과정에서 활용 가능한 다양한 Loss Function 구현 코드입니다.
- `optimizer.py`: SMP 모듈 학습과정에서 활용 가능한 다양한 Optimizer 코드입니다.
- `pseudo_label.py`: 데이터 증강 기법 중 하나인 Pseudo Labelling을 구현해 학습에 활요했습니다.
- `scheduler.py`: SMP 모듈 학습과정에서 활용 가능한 다양한 Scheduler 코드입니다.

### Citing
```
@misc{Iakubovskii:2019,
  Author = {Pavel Iakubovskii},
  Title = {Segmentation Models Pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

### References
- Github: [SMP](https://github.com/qubvel-org/segmentation_models.pytorch)
- Documents: [SMP](https://smp.readthedocs.io/en/latest/index.html)