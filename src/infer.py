import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import glob as glob
import os
from model import build_model
from dataset import normalize_transform
from torchvision import transforms
import yaml
import time
# Constants.
DATA_PATH = '../input/test'
IMAGE_SIZE = 288
NUM_CLASSES = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Class names.
YAML_PATH = '../input/class_dict.yaml'
with open(YAML_PATH, 'r') as file:
    gt_class_dict = yaml.load(file, Loader=yaml.FullLoader)
YAML_PATH = '../input/class_dict_name.yaml'
with open(YAML_PATH, 'r') as file:
    gt_name_dict = yaml.load(file, Loader=yaml.FullLoader)

# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=NUM_CLASSES).to(DEVICE)
checkpoint = torch.load('../outputs/model_pretrained_True.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

# create record
record = pd.DataFrame(columns=['truth_name','truth_num', 'pred_correct', 'avg_confidence'],
                      index= range(NUM_CLASSES))
record['truth_name'] = [gt_class_dict[i] for i in range(100)]
record['truth_num'] = 0
record['pred_correct'] = 0
record['avg_confidence'] = 0

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*")
i = 0
correct = 0
time_prep = []
time_infer = []
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-1].split('.')[0].split('_')[0]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Preprocess the image
    time_start = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained=True)
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)
    time_end = time.time()
    time_prep.append(time_end - time_start)

    # Forward pass throught the image.
    time_start = time.time()
    with torch.no_grad():
        model.eval()
        outputs = model(image)
        prob = nn.LogSoftmax(dim=1)(outputs)
        confidence, preds = torch.max(torch.exp(prob), 1)
    time_end = time.time()
    time_infer.append(time_end - time_start)
    print('image_path: ', image_path)
    print(f'pred_class: ', preds.item())
    print(f'confidence: ', confidence.item())
    gt_id = gt_name_dict[gt_class_name]
    pred_class_name = gt_class_dict[preds.item()]
    print(f"GT: {gt_class_name}, Pred: {pred_class_name}")
    if gt_id == preds.item():
        correct += 1
        record.loc[gt_id, 'pred_correct'] += 1

    # Annotate the image with ground truth.
    cv2.putText(
        orig_image, f"GT: {gt_class_name}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA
    )
    # Annotate the image with prediction.
    cv2.putText(
        orig_image, f"Pred: {pred_class_name.lower()}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (100, 100, 225), 1, lineType=cv2.LINE_AA
    ) 
    cv2.waitKey(0)

    # Save the image with the ground truth class name.
    if not os.path.exists(f"../outputs/inference/{gt_class_name}"):
        os.makedirs(f"../outputs/inference/{gt_class_name}")
    cv2.imwrite(f"../outputs/inference/{gt_class_name}/{gt_class_name}{i}.png", orig_image)
    i += 1
    record.loc[gt_id, 'truth_num'] += 1
    record.loc[gt_id, 'avg_confidence'] += torch.exp(prob)[:,gt_id].item()

print(f"Accuracy: {correct/i}")
print(f"Average time for preprocessing: {1000*np.mean(time_prep)} ms")
print(f"Average time for inference: {1000*np.mean(time_infer)} ms")
record['avg_confidence'] = record['avg_confidence'] / record['truth_num']
record.to_csv('../outputs/infer_record.csv', index=False)