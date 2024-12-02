# Hand Bone Image Segmentation

## **📘**Overview

2024.11.13 ~ 2024.11.28

This project focuses on segmenting hand bones in medical images as part of a private competition organized by Naver Connect Foundation and Upstage.


## **📘**Contributors

|은의찬|임동훈|김예나|한승수|김동영|정아영
|:----:|:----:|:----:|:----:|:----:|:----:|
| [<img src="https://github.com/user-attachments/assets/de2fa83d-3076-4f18-bc65-45e34a456b72" alt="" style="width:100px;100px;">](https://github.com/0522chan) <br/> | [<img src="https://github.com/user-attachments/assets/6ba55701-35e6-421f-8ed7-03b054f55a76" alt="" style="width:100px;100px;">](https://github.com/naringles) <br/> | [<img src="https://github.com/user-attachments/assets/109315cf-03ea-46c9-af2d-4145cef1f50f" alt="" style="width:100px;100px;">](https://github.com/yehna2907) <br/> | [<img src="https://github.com/user-attachments/assets/b2e040a7-dca3-4a23-b44f-5de84b76c950" alt="" style="width:100px;100px;">](https://github.com/hanseungsoo13) <br/> | [<img src="https://github.com/user-attachments/assets/d973c9de-7e57-4796-8c48-924269f4d2c9" alt="" style="width:100px;100px;">](https://github.com/kimdyoc13) <br/> | [<img src="https://github.com/user-attachments/assets/1a023730-0169-427f-8642-977aa888535e" alt="" style="width:100px;100px;">](https://github.com/Jeong-AYeong) <br/> |


## **📘**Wrap up Report

[Here's our link](https://broadleaf-jasper-0c4.notion.site/d7a2c94d5c604e8380479662a227c8b0)

## **📘**Metrics

- Dice

![스크린샷 2024-12-01 215355](https://github.com/user-attachments/assets/0a4b33ba-0901-486c-963d-ddabada68fe2)



## **📰**Tools

- github
- notion
- slack
- wandb

## **📰**Folder Structure

```

├─code
│  └─baseline_code.ipynb
├─mmsegmentation
│  ├─custom_config
|  |  ├─data_vars.py
|  |  ├─dataset_setting.py
|  |  ├─default_runtime.py
|  |  └─segformer.py
│  └─custom_modules
|    ├─datasets
|    |  ├─__init__.py
|    |  └─custom_dataset.py
|    ├─metrics
|    |  ├─__init__.py
|    |  └─custom_metric.py
|    ├─models
|    |  ├─__init__.py
|    |  └─custom_model.py
|    ├─transforms
|    |  ├─__init__.py
|    |  └─custom_transform.py
├─ensemble
│  ├─ensemble.py
|  └─soft_voting_setting.yaml
```

## **📰**Dataset Structure

```

├─data
     ├─test
     │    └─DCM
     │         ├─ID040
     │         │     image1661319116107.png
     │         │     image1661319145363.png
     │         └─ID041
     │                image1661319356239.png
     │                image1661319390106.png
     │
     ├─train
     │    ├─DCM
     │    │   ├─ID001
     │    │   │     image1661130828152_R.png
     │    │   │     image1661130891365_L.png
     │    │   └─ID002
     │    │          image1661144206667.png
     │    │          image1661144246917.png
     │    │        
     │    └─outputs_json
     │               ├─ID001
     │               │     image1661130828152_R.json
     │               │     image1661130891365_L.json
     │               └─ID002
                             image1661144206667.json
                             image1661144246917.json
```

- images : 1088
    - train : 800
    - test : 288
- 29 class : f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, Trapezium, Trapezoid, Capitate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform, Radius, Ulna
- image size :  (2048, 2048)

![스크린샷 2024-12-01 215433](https://github.com/user-attachments/assets/8a3a4c59-0ad8-447b-9315-a964b86de361)


## **📰**Model (backbone)
- FCN (torchvision)
- DeeplabV3+ (xception71)
- UNet++ (xception)
- UNet++ (effinet-b5)
- UNet++ (resnext101)
- UNet++ (hrnet_64)
- UNet++ (max-vit)
- UNet++ (effinetv2-xl)
- UNet++ (halonet50)
- UNet++ (nfnet_l2)

## **📰**Encoder 
- ResNet
- ResNeXt
- ResNeSt
- RegNet
- GERNet
- EfficientNet
- MobileNet
- VGG19

## **📰Experiments**
![스크린샷 2024-12-01 214215](https://github.com/user-attachments/assets/02200029-5ca1-441a-a637-6269bfc83905)


| Exp | mDICE |
| --- | --- |
| Unet++_hrnet_5fold | 0.9741 |
| Unet++hrnet, Unet++vgg, segformer, deeplab | 0.9751 |
| Unet++hrnet, Unet++vgg, segformer, deeplab(threshold:0.4) | 0.9753 |



