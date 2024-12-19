import torch
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter

# Required constants.
ROOT_DIR = '/home/data/Dataset/matches_all'
VALID_SPLIT = [0.7,0.2,0.1]
IMAGE_SIZE = 288 # Image size of resize when applying transforms. 224 for b0, 528 for b6
BATCH_SIZE = 32
NUM_WORKERS = 8 # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

def get_datasets(pretrained):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(get_train_transform(IMAGE_SIZE, pretrained))
    )
    dataset_test = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    dataset_size = len(dataset)
    # Calculate the validation dataset size.
    train_size = int(dataset_size * VALID_SPLIT[0])
    valid_size = int(dataset_size * VALID_SPLIT[1])
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:train_size])
    dataset_valid = Subset(dataset_test, indices[train_size:(train_size+valid_size)])
    dataset_test = Subset(dataset_test, indices[(train_size+valid_size):])
    # Get training sample distribution.
    train_stats = dict(Counter(dataset_train.dataset.targets))
    class_weights = torch.tensor([1/train_stats[i] for i in range(len(dataset.classes))])
    # Save label-index mapping.
    with open('../input/class_dict.yaml', 'w') as file:
        yaml.dump(dict((v,k.split('.')[-1]) for k,v in dataset.class_to_idx.items())
                  , file) # index to name
    with open('../input/class_dict_name.yaml', 'w') as file:
        yaml.dump(dict((k.split('.')[-1],v) for k, v in dataset.class_to_idx.items())
                  , file) # name to index 
    return dataset_train, dataset_valid, dataset_test, dataset.classes, class_weights

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader

if __name__ == '__main__':
    # get label index mapping
    dataset = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(get_train_transform(IMAGE_SIZE, True))
    )
   
    # get dataset statistics
    class_dict = dict((v,k.split('.')[-1]) for k,v in dataset.class_to_idx.items())
    file = open('../dataset_stats.yaml', 'w')
    stats = dict(Counter(dataset.targets))
    stats = dict((class_dict[k], v) for k,v in stats.items())
    yaml.dump(stats, file)
    file.close()