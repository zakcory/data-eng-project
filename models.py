from dataclasses import dataclass
import torch_geometric
from torch_geometric.nn import SAGEConv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.optim



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
    
# training function for the GNN model
def train_gnn_model(model, data, cfg):

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    loss_steps = list()
    best_val_acc = 0
    best_loss = np.inf

    for epoch in range(1, cfg.epochs+1):

        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        val_acc = accuracy(logits[data.valid_mask], data.y[data.valid_mask])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_loss = loss.item()
            torch.save({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, cfg.ckpt_dir)
    
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}  "
                  f"Best Val Acc: {best_val_acc:.4f}  "
                  f"Best Loss: {best_loss:.4f}  "
            )
        loss_steps.append(loss.item())
    return loss_steps, model

    

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

class GlassNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GlassNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=1)
        return x

def train_deep_model(model, x_train, y_train, x_val, y_val, cfg):
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()


    loss_steps = list()
    best_val_acc = 0
    best_loss = np.inf

    n_batches = int(np.ceil(len(x_train) / cfg.batch_size)) 
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch_idx in range(n_batches):
                start_idx = batch_idx * cfg.batch_size
                end_idx = min((batch_idx + 1) * cfg.batch_size, len(x_train))

                x = x_train[start_idx:end_idx, :].detach().clone().to(cfg.device)
                y = y_train[start_idx:end_idx].detach().clone().to(cfg.device)
                output = model.forward(x)    # scores 

                optimizer.zero_grad()
                loss = criterion(output, y)        
                loss.backward()
                optimizer.step()

        val_acc = validate(model, x_val, y_val, cfg.batch_size, cfg.device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_loss = loss.item()
            torch.save({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, f"cfg.ckpt_dir")
    
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}  "
                  f"Best Val Acc: {best_val_acc:.4f}  "
                  f"Best Loss: {best_loss:.4f}  "
            )
        loss_steps.append(loss.item())

    return loss_steps, model


class TrainConfig:
    def __init__(self, epochs, lr, weight_decay, momentum, batch_size, ckpt_dir, log_every, device):
       self.epochs = epochs 
       self.lr = lr
       self.weight_decay = weight_decay
       self.momentum = momentum
       self.batch_size = batch_size
       self.ckpt_dir = ckpt_dir
       self.log_every = log_every
       self.device = device
    def __repr__(self):
        print(f"Train Config: epochs={self.epochs}, lr={self.lr}, bs={self.batch_size}, log every {self.log_every}")



# useful functions for models

@torch.no_grad()
def accuracy(logits, y):
    pred = logits.argmax(dim=11)
    y = y.view(-1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def validate(model, x_val, y_val, bs, device):
    model.eval()
    correct, total = 0, 0
    n_batches = int(np.ceil(len(x_val) / bs)) 
    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min((batch_idx + 1) * bs, len(x_val))
        x = x_val[start_idx:end_idx, :].detach().clone().to(device)
        y = y_val[start_idx:end_idx].detach().clone().to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total