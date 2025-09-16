from dataclasses import dataclass
from torch_geometric.nn import SAGEConv
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models



# TODO: ADD MODELS HERE

# GNN model (for active learning)
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv2 = SAGEConv(in_channels=hidden_channels, out_channels=output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x
    

# CNN model (for CIFAR10)
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
            super().__init__()
            # load resnet18 without pretrained weights
            net = models.resnet18(weights=None)

            # adjust first layers 
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

            self.layers = nn.Sequential(
                net.conv1, net.bn1, net.relu,
                net.layer1, net.layer2, net.layer3, net.layer4,
                net.avgpool
            )
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        
        # for the AL pipeline 
        if return_embedding:
            return x
        return self.fc(x)




# loading model function
def load_model_wrapper(model_name, n_classes):
    factory = {
        "resnet18": ResNet18(n_classes),
        "gcn": GraphSAGE(n_classes)
        # add every model that gets added here
    }
    if model_name not in factory:
        raise ValueError(f"Unknown model {model_name}")
    return factory[model_name]

# training model function
def train_model_wrapper(X_train, y_train, model):
    pass


class TrainConfig:
    def __init__(self, epochs, lr, weight_decay, momentum, batch_size, ckpt_dir, log_every):
       self.epochs = epochs 
       self.lr = lr
       self.weight_decay = weight_decay
       self.momentum = momentum
       self.batch_size = batch_size
       self.ckpt_dir = ckpt_dir
       self.log_every = log_every
    def __repr__(self):
        print(f"Train Config: epochs={self.epochs}, lr={self.lr}, bs={self.batch_size}, log every {self.log_every}")