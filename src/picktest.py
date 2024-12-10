import glob as glob
import os
import cv2
import random
from datetime import datetime
DATA_PATH = '../input/matches2'
SAMPLE_PER_CLASS = 5
# pick 5 random images from each subdirectories of dataset.
all_image_paths = glob.glob(f"{DATA_PATH}/*")
for image_path in all_image_paths:
    # get random files
    all_files = os.listdir(image_path)
    gt_class_name = image_path.split(os.path.sep)[-1].split('.')[1]
    random.seed(int(datetime.now().timestamp()))
    index = random.sample(range(0, len(all_files)), SAMPLE_PER_CLASS)
    counter = 0
    if not os.path.exists('../input/test'):
        os.makedirs('../input/test')
    for i in index:
        img = cv2.imread(os.path.join(image_path, all_files[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_image = img.copy()
        cv2.imwrite(f'../input/test/{gt_class_name}_{counter}.png', orig_image)
        counter += 1
    