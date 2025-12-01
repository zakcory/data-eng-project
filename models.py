"""
Model architectures and training utilities for GNN-based Active Learning.
Includes CNN (ResNet18), LSTM, GNN (GraphSAGE), and MLP models.
"""

from dataclasses import dataclass
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.optim
from copy import deepcopy
from torch.optim.lr_scheduler import LinearLR


class SentimentLSTM(nn.Module):
    """
    LSTM model for text sentiment classification (IMDB dataset).
    
    Args:
        vocab_size: Size of vocabulary
        output_size: Number of output classes (2 for binary sentiment)
        embedding_dim: Dimension of word embeddings
        hidden_dim: LSTM hidden state dimension
        n_layers: Number of LSTM layers
        drop_prob: Dropout probability
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Stack up lstm outputs
        lstm_out = lstm_out[:, -1, :]

        out = self.dropout(lstm_out)
        out = self.fc(out)

        # Return last sigmoid output and hidden state
        return out, hidden, lstm_out

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        device = next(self.parameters()).device
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

        return (h0, c0)


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for label propagation in active learning.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        output_dim: Number of output classes
    """
        
    def __init__(self, in_channels, hidden_channels, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = SAGEConv(in_channels=hidden_channels, out_channels=output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        return x


def train_gnn_model(model, data, loader, cfg, patience=50):
    """
    Train GraphSAGE model with early stopping.
    
    Args:
        model: GraphSAGE model
        data: PyG data object
        loader: NeighborLoader for mini-batch training
        cfg: Training configuration
        patience: Early stopping patience
    
    Returns:
        loss_steps: Training loss history
        best_model: Best model based on validation accuracy
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    loss_steps = list()
    best_val_acc = 0
    best_loss = np.inf
    best_model = None
    patience_count = 0

    for epoch in range(1, cfg.gnn_epochs+1):

        # Early stopping check
        if patience_count > patience:
            print(f"Breaking early... (patience)")
            break
        model.train()

        # Mini-batch training
        for batch in loader:
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

        # Validation
        val_acc, _ = validate_gnn(model, data, data.valid_mask)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_loss = loss.item()
            best_model = deepcopy(model)
            patience_count = 0
        else:
            patience_count += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}  "
                  f"Best Val Acc: {best_val_acc:.4f}  "
                  f"Best Loss: {best_loss:.4f}  "
            )
        loss_steps.append(loss.item())
    return loss_steps, best_model


class ResNet18(nn.Module):
    """
    Modified ResNet18 for CIFAR-10 (32x32 images).
    Adjusts first conv layer and removes maxpool for smaller images.
    """

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
        """
        Forward pass with optional embedding extraction.
        
        Returns:
            logits: Class predictions
            inter_emb: Feature embeddings before final FC layer
        """
                
        x = self.layers(x)
        inter_emb = torch.flatten(x, 1)

        return self.fc(inter_emb), inter_emb


class BeanNet(nn.Module):
    """
    2-layer MLP for DryBean tabular dataset.
    
    Args:
        input_dim: Number of input features (16 for DryBean)
        num_classes: Number of bean types (7)
    """

    def __init__(self, input_dim, num_classes):
        super(BeanNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """Forward pass returning both predictions and embeddings."""
        inter_emb = torch.relu(self.fc1(x))
        x = self.fc3(inter_emb) 
        return x, inter_emb


def train_deep_model(model, x_train, y_train, x_val, y_val, cfg, fine_tune, first_round, patience=40):
    """
    Train base models (CNN/LSTM/MLP) with optional fine-tuning.
    
    Args:
        model: Model to train
        x_train, y_train: Training data
        x_val, y_val: Validation data
        cfg: Training configuration
        fine_tune: Whether to fine-tune (fewer epochs)
        first_round: Is this the first AL iteration
        patience: Early stopping patience
    
    Returns:
        loss_steps: Training loss history
        best_model: Best model based on validation accuracy
    """

    model = model.to(cfg.device)
    print("Model on GPU")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Adjust epochs for fine-tuning
    if (fine_tune and not first_round):
        epochs = 50 if cfg.model_name == 'resnet18' else 10
        # Model-specific fine-tuning epochs
        if cfg.model_name == 'resnet18':
            50
        elif cfg.model_name == 'beannet':
            1
        elif cfg.model_name == 'lstm':
            10

        # reinitialize the linear head for CNN
        if cfg.model_name == 'resnet18': reinit_fc_head(model)
    else:
        epochs = cfg.epochs
    
    # Learning rate schedule for fine-tuning
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=0.5*epochs) 
    
    loss_steps = list()
    best_val_acc = 0.
    patience_count = 0
    best_loss = np.inf
    best_model = None

    n_batches = int(np.ceil(len(x_train) / cfg.batch_size))
    for epoch in range(1, epochs + 1):

        if patience_count > patience:
            print(f"Breaking early... (patience)")
            break

        model.train()

        # Mini-batch training
        for batch_idx in range(n_batches):
                start_idx = batch_idx * cfg.batch_size
                end_idx = min((batch_idx + 1) * cfg.batch_size, len(x_train))

                x = x_train[start_idx:end_idx, :].detach().clone().to(cfg.device)
                y = y_train[start_idx:end_idx].detach().clone().to(cfg.device)

                if cfg.model_name == 'lstm':
                    # LSTM-specific handling
                    current_batch_size = x.size(0)
                    h = model.init_hidden(current_batch_size)
                    h = tuple(s.detach() for s in h)  #Detach hidden state
                    model.lstm.flatten_parameters()   # Optimize LSTM memory layout

                    y = y.long()
                    output, h, _ = model(x, h)
                else:
                    output, _ = model(x)   

                optimizer.zero_grad()
                loss = criterion(output, y)
                loss.backward()

                optimizer.step()

        # Step scheduler for fine-tuning
        if fine_tune and not first_round:
            scheduler.step()

        # Validation
        val_acc, _, _ = validate(model, x_val, y_val, cfg.batch_size, cfg.device)

        # Save best model
        if val_acc > best_val_acc:
            patience_count = 0
            best_val_acc = val_acc
            best_loss = loss.item()
            best_model = deepcopy(model)
        else:
            patience_count += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}  "
                  f"Best Val Acc: {best_val_acc:.4f}  "
                  f"Best Loss: {best_loss:.4f}  "
            )
        loss_steps.append(loss.item())

    # Recalibrate BatchNorm for ResNet after fine-tuning
    if cfg.model_name == 'resnet18' and fine_tune and not first_round:
        recalibrate_bn(best_model, x_train, cfg)

    return loss_steps, best_model

@dataclass
class TrainConfig:
    """Training configuration container."""
    epochs: int
    gnn_epochs: int
    lr: float
    weight_decay: float
    momentum: float
    batch_size: int
    log_every: int
    device: str
    model_name: str
    def __repr__(self):
        return f"Train Config: epochs={self.epochs}, lr={self.lr}, bs={self.batch_size}, log every {self.log_every}"


# Validation and utility functions
@torch.no_grad()
def validate_gnn(model, data, mask):
    """
    Validate GNN model using neighbor sampling.
    
    Args:
        model: GraphSAGE model
        data: PyG data object
        mask: Boolean mask for nodes to evaluate
    
    Returns:
        acc: Accuracy on masked nodes
        all_probs: Prediction probabilities
    """

    model.eval()
    # Create loader for validation nodes
    loader = NeighborLoader(
        data,
        input_nodes=mask,
        num_neighbors=[-1],  # full neighborhood for val nodes
        batch_size=256,
        shuffle=False
    )
    
    all_probs, all_preds, all_labels = [], [], []

    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        out = model(batch.x, batch.edge_index)

        # Get outputs/labels only for the "seed" nodes (from the mask)
        out_seed = out[:batch.batch_size]
        y_seed = batch.y[:batch.batch_size]

        probs = F.softmax(out_seed, dim=1)  # Use out_seed
        preds = probs.argmax(dim=1)
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(y_seed.detach().cpu())  # Use y_seed

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()
    return acc, all_probs


@torch.no_grad()
def validate(model, x_val, y_val, bs, device):
    """
    Validate base models (CNN/LSTM/MLP).
    
    Returns:
        accuracy: Validation accuracy
        probs: Prediction probabilities
        embeddings: Feature embeddings
    """
    model.eval()
    correct, total = 0, 0
    n_batches = int(np.ceil(len(x_val) / bs))
    probs_stack = list()
    emb_stack = list()

    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min((batch_idx + 1) * bs, len(x_val))

        x = x_val[start_idx:end_idx, :].detach().clone().to(device)
        y = y_val[start_idx:end_idx].detach().clone().to(device)

        if isinstance(model, SentimentLSTM):
            h = model.init_hidden(x.size(0))
            logits, _, inter_emb = model(x, h)
        else:
            logits, inter_emb = model(x)

        emb_stack.append(inter_emb.detach().cpu())
        probs = F.softmax(logits, dim=1)
        probs_stack.append(probs.detach().cpu())
        correct += (probs.argmax(dim=1) == y).sum().item()
        total += y.numel()

    return (correct / total), torch.cat(probs_stack, dim=0), torch.cat(emb_stack, dim=0)


def recalibrate_bn(model, x_full, cfg):
    """
    Recalibrate BatchNorm statistics after fine-tuning.
    Runs forward passes to update running stats.
    """
    was_training = model.training
    model.train()
    bs = cfg.batch_size
    device = cfg.device
    n_batches = int(np.ceil(len(x_full) / bs))
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, len(x_full))
            x = x_full[start_idx:end_idx, :].detach().clone().to(device)
            model(x)
            if batch_idx >= 50:  
                break
    model.train(was_training)


def reinit_fc_head(model):
    """Reinitialize final FC layer for better fine-tuning."""
    head = model.fc
    nn.init.kaiming_normal_(head.weight, nonlinearity='linear')
    if head.bias is not None:
        nn.init.zeros_(head.bias)