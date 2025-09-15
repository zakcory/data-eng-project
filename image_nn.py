from __future__ import annotations
import os
import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models



# ----------------------------
# Training / Evaluation
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, return_embedding=False)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs, return_embedding=False)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ----------------------------
# Embedding extraction
# ----------------------------
@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        embeddings: (N, 512)
        labels:     (N,)
    Set max_batches to limit for quick tests (None = all).
    """
    model.eval()
    embs: List[torch.Tensor] = []
    lbls: List[torch.Tensor] = []
    for b_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs, return_embedding=True)  # (B, 512)
        embs.append(emb.cpu())
        lbls.append(labels.clone())
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break
    return torch.cat(embs, dim=0), torch.cat(lbls, dim=0)


# ----------------------------
# Save / Load
# ----------------------------
def save_weights(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model: nn.Module, path: str, device: torch.device) -> None:
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    model = ResNet18WithEmbedding(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10
    best_acc = 0.0
    weights_path = "checkpoints/cifar10_resnet18.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}: loss={train_loss:.4f}  test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_weights(model, weights_path)
            print(f"  ↳ Saved weights to {weights_path}")

    # --- Example: Load best weights and extract embeddings from the test set ---
    print("Reloading best weights and extracting embeddings from test set...")
    load_weights(model, weights_path, device)
    embeddings, labels = extract_embeddings(model, test_loader, device, max_batches=None)
    print(f"Embeddings shape: {embeddings.shape}  Labels shape: {labels.shape}")

    # Optional: persist embeddings for later use
    torch.save(
        {"embeddings": embeddings, "labels": labels},
        "checkpoints/cifar10_test_embeddings.pt",
    )
    print("Saved embeddings to checkpoints/cifar10_test_embeddings.pt")


if __name__ == "__main__":
    main()
