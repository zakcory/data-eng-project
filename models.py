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
class ResNet18Simple(nn.Module):
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
def load_model_wrapper(model_name):
    # example usage
    if model_name == 'ResNet140':
        pass

# training model function
def train_model_wrapper(X_train, y_train, model):
    pass