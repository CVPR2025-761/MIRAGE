# Implementation of MIRAGE: Medical Image-Text Pre-training for Robustness Against Noisy Environments


### Setup
Run 

```bash
pip install -r requirements.txt
```


###  Data Preparation

#### PMC-OA Dataset

1. Download PMC-OA from `https://github.com/WeixiongLin/PMC-CLIP/`, including one image zip (image.zip), three jsonl files (train.jsonl, valid.jsonl, test,jsonl).

2. Unzip image.zip and mark the output dir as `<image_path>`

3. Add `<image_path>` to `configs/data/pmc.yaml`

```yaml
_target_: mirage.data.pretrain.pmc.pmc_oa.PmcOaDataset
root_path: <image_path> # update this line
dataset_path: ???
image_transform: ???
text_transform: []
num_colors: 3
rate: 1.0
max_length: 77
pretrained_name: ${pretrain_model.text_encoder.pretrained_name}
```

4. Add three jsonls files to `configs/experiments/pre_train/train_mirage`


```yaml
    train_dataset:
      dataset_path: <train_jsonl_path> # update this line
      image_transform:
        - ${image_transformation.resize}
        - ${image_transformation.random_affine}
        - ${image_transformation.random_horizontal_flip}
      rate: 1.0
    validation_dataset:
      dataset_path: <valid_jsonl_path> # update this line
      image_transform:
        - ${image_transformation.resize}
      rate: 1.0
    test_dataset:
      dataset_path: <test_jsonl_path> # update this line
      image_transform:
        - ${image_transformation.resize}
      rate: 1.0
```

### Model Preparation
We released the pre-trained ResNet50 weights pre-trained on PMC-OA-Image set using MoCo at [release page](https://github.com/CVPR2025-761/MIRAGE/releases/download/Weights/resnet50_moco.ckpt). Please put it in the `./pretrained/resnet50_moco.pt`. MIRAGE will firstly load the weights into ResNet50 automatically before training.


### Pre-train
Run
```bash
cd codes/
python scripts/pre_train.py +experiments/pre_train=train_mirage
```


### Pre-trained weights
We have provided the pre-trained MIRAGE-ResNet50 weights at [release page](https://github.com/CVPR2025-761/MIRAGE/releases/download/Weights/mirage.ckpt). We plan to release the pre-trained weights and MIRAGE with more backbone architectures (ResNet101, ResNeSt, ViT, Swin, etc.).


### Downstream tasks
This part is under code review, and will be uploaded soon.

