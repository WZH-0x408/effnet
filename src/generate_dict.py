from torchvision import datasets, transforms
import yaml

ROOT_DIR = '../input/matches2'
dataset = datasets.ImageFolder(ROOT_DIR, 
                               transform=transforms.ToTensor())
with open('../input/class_dict.yaml', 'w') as file:
    yaml.dump(dict((v,k.split('.')[-1]) for k,v in dataset.class_to_idx.items()), 
              file) # index to name
with open('../input/class_dict_name.yaml', 'w') as file:
    yaml.dump(dict((k.split('.')[-1],v) for k, v in dataset.class_to_idx.items()),
              file) # name to index 