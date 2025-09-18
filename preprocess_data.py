# Here we will take a csv/any dataset and transform it into X (set of datapoints) and y (set of labels for each point)
# Then, we will pass it to the pipeline in this form

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from thinc.util import to_categorical
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd

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


def get_glass_data(data_dir="./data/glass/glass.csv"):
    """
    Loads and preprocesses the entire glass dataset.
    """
    df = pd.read_csv(data_dir)

    # 1. Separate features (X) and the target label (y)
    X = df.drop('Type', axis=1).values
    y = df['Type'].values

    # 2. Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_tensor = torch.from_numpy(X_scaled).float()
    y_tensor = torch.from_numpy(y_encoded).long()

    return X_tensor, y_tensor

def get_imdb_data(data_dir):
    pass