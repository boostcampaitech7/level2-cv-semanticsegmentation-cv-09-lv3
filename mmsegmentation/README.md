# MMsegmentation
The mmsegmentation directory contains the core modules and configuration files for **semantic segmentation** in this project.
This README provides an overview of the directory structure, training, and inference commands.

## **🚀**Getting Started
### MMEngine / MMCV Installation
```
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### MMSegmentation 
```
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## **📂**Directory Structure
Organize the cloned `mmsegmentation` repository according to the following directory structure. 
This includes adding custom configurations, modules, and tools to extend its functionality:
```
mmsegmentation/
├──configs/
│  ├── ...
│  ├──custom_config
│  │  ├──segformer.py        # Config file for SegFormer
│  │  ├──data_vars.py        # Dataset-related variables (directory path, etc.)
│  │  ├──dataset_setting.py  # Dataloader settings
│  │  ├──default_runtime.py  # Wandb settings
│   ...
├──custom_modules/
│  ├──datasets
│  │  ├──__init__.py
│  │  ├──custom_dataset.py
│  ├── metrics
│  │  ├──__init__.py
│  │  ├──custom_metric.py
│  ├── models
│  │  ├──__init__.py
│  │  ├──custom_model.py
│  ├── transforms
│  │  ├──__init__.py
│  │  ├──custom_transform.py
├── tools/
│  ├──train.py
│  ├──inference.py
│   ...
```
- `configs/custom_config/`: Contains custom configuration files for models, datasets, and runtime settings.
- `custom_modules/`: Includes custom implementations for datasets, metrics, models, and data transforms.
- `tools/`: Contains training and inference scripts.


## **📘**Train / Inference
### Training
```
# python tools/train.py {CONFIG}
python tools/train.py configs/custom_config/segformer.py
```
### Inference
```
# python tools/inference.py {CONFIG} {CHECKPOINT}
python tools/inference.py configs/custom_config/segformer.py work_dir/.../last_checkpoint
```