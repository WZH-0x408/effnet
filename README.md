## Introduction
This is a reproduction of the fine-tuning procedure of EfficientNet, using packed-up models offered by [torchvision], referencing largely the excellent work of [Sovit Rajan Rath] at [Efficientnet Transfer Learning Reference], with tweaks on handling larger datasets and visualization options.

[Efficientnet Transfer Learning Reference]: https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
[Sovit Rajan Rath]: <https://debuggercafe.com/>
[torchvision]:<https://pytorch.org/vision/main/models/efficientnet.html>

## Reproduce the code
#### 1. Prepare the Data
The dataset is organized in [PyTorch ImageFolder] fashion, in that every sub-directory is named after the corresponding label. You may have to change the ROOT_DIR to your dataset directory in [dataset.py](src/dataset.py).

#### 2. Prepare the Labels
Run [generate_dict.py](src/generate_dict.py) to generate mappings between label names and indices. This is used for visualization only during inference. You may have to change the ROOT_DIR to load the data.

#### 3. Train
Run [train.py](src/train.py) to fine-tune the model and generate weights. You may have to change the [IMAGE_SIZE] hyperparameter for different models.
```
python train.py --epochs 20 --pretrained --model-type efficientnet_b6
```

#### 4. Inference
Run [picktest.py](src/model.py) to generate test directory for observing inference performance. You may have to change the DATA_PATH to your dataset directory .

Run [infer.py](src/infer.py) to obtain images annotated with inference results under [outputs/inference](outputs/inference). Recorded performance is at [infer_record.csv](outputs/infer_record.csv).

[PyTorch ImageFolder]:<https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html>
[IMAGE_SIZE]:<https://pytorch.org/vision/main/models/efficientnet.html>
