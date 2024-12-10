import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, fine_tune=True, num_classes=10, model_type='efficientnet_b2'):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    # model = models.efficientnet_b0(pretrained=pretrained)
    if model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_type == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained)
    elif model_type == 'efficientnet_b6':
        model = models.efficientnet_b6(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    if model_type == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    elif model_type == 'efficientnet_b2':
        model.classifier[1] = nn.Linear(in_features=1408, out_features=num_classes)
    elif model_type == 'efficientnet_b6':
        model.classifier[1] = nn.Linear(in_features=2304, out_features=num_classes)
        
    return model 

if __name__ == '__main__':
    model = build_model(pretrained=True, fine_tune=False, num_classes=10)
    print(model)
