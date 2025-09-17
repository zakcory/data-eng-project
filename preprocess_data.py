# Here we will take a csv/any dataset and transform it into X (set of datapoints) and y (set of labels for each point)
# Then, we will pass it to the pipeline in this form

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# CIFAR10 loader
def get_cifar10_data(data_dir="./data/cifar10"):

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    # we need the entire dataset for tensor indexing later on 
    X_train = torch.from_numpy(train_set.data)
    X = torch.cat([X_train, torch.from_numpy(test_set.data)], dim=0)  
    y = torch.tensor(train_set.targets + test_set.targets, dtype=torch.long)

    # normalizing the tensor
    X = X.permute(0, 3, 1, 2).float().div_(255.0)
    mean = X.mean(dim=(0, 2, 3), keepdim=True)              
    std  = X.std(dim=(0, 2, 3), keepdim=True)  
    print(mean, std)
    X = (X - mean) / std

    return X, y

def get_credit_data(data_dir):
    pass

def get_imdb_data(data_dir):
    pass