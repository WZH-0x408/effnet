import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from dataset import get_datasets, get_data_loaders
from utils import save_model, save_plots, save_best

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.0001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-m', '--model-type', type=str, default='efficientnet_b2',
    dest='model',
    help='Type of model to train, including efficientnet_b0, efficientnet_b2, efficientnet_b6'
)
args = vars(parser.parse_args())

# Training function.
def train(model, trainloader, optimizer, criterion, num_classes, device='cuda'):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    conf_matrix = torch.zeros(num_classes, num_classes)
    class_stat = torch.zeros(num_classes, 1)
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy and confusion matrix.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        for t, p in zip(labels.view(-1), preds.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
            class_stat[t.long()] += 1
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, conf_matrix, class_stat

# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# test function.
def test(model, testloader):
    model.eval()
    print('Test')
    valid_running_correct = 0
    counter = 0
    # load class dict
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the accuracy.
            confidence, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    # Loss and accuracy for the complete epoch.
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_acc

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test, dataset_classes, class_weights = get_datasets(args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    print(f'[INFO]: Number of classes: {len(dataset_classes)}')
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    model_type = args['model']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Model type: {model_type}")
    print(f"Epochs to train for: {epochs}\n")
    model = build_model(
        pretrained=args['pretrained'], 
        fine_tune=True, 
        num_classes=len(dataset_classes),
        model_type=model_type
    ).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_val_loss = float('inf')
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, conf_matrix, class_stat = train(model, train_loader, 
                                                                           optimizer, criterion,
                                                                           num_classes=len(dataset_classes),
                                                                           )
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        # record the best model in terms of val loss.
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            best_model_state = model.state_dict()
            best_optim_state = optimizer.state_dict()
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, args['pretrained'])
    # Save the best model weights.
    save_best(epochs, best_model_state, best_optim_state, criterion, args['pretrained'])
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, args['pretrained'])
    # Save the confusion matrix.
    conf_matrix = conf_matrix.cpu().numpy()
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    conf_matrix = conf_matrix
    df_cm = pd.DataFrame(conf_matrix, 
                         range(len(dataset_classes)), 
                         range(len(dataset_classes)))
    df_cm.to_csv('../outputs/confusion_matrix.csv')
    plt.figure()
    sns.set_theme(rc={'figure.figsize':(60,50)})
    sns.heatmap(df_cm, annot=False, cmap='viridis', fmt='g')
    plt.savefig('../outputs/confusion_matrix.png')
    print('TRAINING COMPLETE')

    # Test the model.
    _, test_loader = get_data_loaders(dataset_train, dataset_test)
    acc = test(model, test_loader)
    print(f"Test accuracy: {acc:.3f}")