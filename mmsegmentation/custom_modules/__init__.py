from .models import *
from .datasets import *
from .transforms import *
from .metrics import *
from mmseg.registry import MODELS, DATASETS, TRANSFORMS, METRICS

# Registry 초기화 함수
def register_custom_modules():
    import custom_modules.models
    import custom_modules.datasets
    import custom_modules.transforms
    import custom_modules.metrics
    # Registry에 필요한 초기화 작업을 여기서 수행 