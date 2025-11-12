import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

import networkx as nx
from sklearn.manifold import TSNE


# (Make sure os, np, torch, and matplotlib.pyplot as plt are also imported in utils.py)

def generate_plot(accuracy_scores_dict, seed, dataset_name):
    num_iters = len(accuracy_scores_dict['random'])
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        x_vals = list(range(1, len(accuracy_scores) + 1))
        plt.plot(x_vals, accuracy_scores, label=criterion)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.xlim(1, num_iters)
    plt.xticks(range(1, num_iters + 1))
    plt.legend()
    plt.title(f'AL - Accuracy vs. Iterations, Seed: {seed}, Dataset: {dataset_name}, FT')
    plt.savefig(f'plot_{seed}_{dataset_name}_FT')
    plt.close()

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_knn_graph(embeddings: torch.Tensor, k: int = 10, symmetrize: bool = True) -> torch.LongTensor:
    """
    Build a k-NN graph from embeddings.

    Args:
        embeddings: torch.Tensor [N, D] (CPU or GPU). If on GPU, ensure enough memory.
        k: number of neighbors per node.
        symmetrize: if True, add reverse edges to make the graph undirected.

    Returns:
        edge_index: LongTensor [2, E] suitable for PyG.
    """
    X = embeddings
    if X.device.type != "cpu":
         #optional: for large N you may want to move to CPU to save GPU mem
         X = X.detach().cpu()

    N = X.size(0)
    # cosine normalization
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)

    chunk = 2048  # adjust for memory
    src_all, dst_all = [], []

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        S = Xn[start:end] @ Xn.t()           # [c, N] cosine similarity
        S[:, start:end].fill_diagonal_(-1.0) # avoid self as top-1
        _, nn_idx = torch.topk(S, k=k, dim=1)

        rows = torch.arange(start, end).unsqueeze(1).expand(-1, k)
        src_all.append(rows.reshape(-1))
        dst_all.append(nn_idx.reshape(-1))

    src = torch.cat(src_all).long()
    dst = torch.cat(dst_all).long()

    if symmetrize:
        src = torch.cat([src, dst], dim=0)
        dst = torch.cat([dst, src[:len(dst)]], dim=0)

    edge_index = torch.stack([src, dst], dim=0)
    # remove any accidental self-loops
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]

def plot_gnn_neighborhood(
    graph_data,
    all_masks,
    node_idx,
    iteration,
    seed,
    dataset_name
):
    """
    Plots the 2-hop neighborhood of a specific node.
    """
    print(f"Plotting 2-hop neighborhood for node {node_idx}...")

    # --- 1. Find 2-hop neighborhood ---
    edge_index = graph_data.edge_index.cpu()
    node_idx = int(node_idx)

    # Get direct neighbors (1-hop)
    mask_to = (edge_index[0] == node_idx)
    mask_from = (edge_index[1] == node_idx)
    neighbors_1 = torch.cat([edge_index[1, mask_to], edge_index[0, mask_from]]).unique()

    # Get 2-hop neighbors (neighbors of neighbors)
    mask_to_2 = torch.isin(edge_index[0], neighbors_1)
    mask_from_2 = torch.isin(edge_index[1], neighbors_1)
    neighbors_2 = torch.cat([edge_index[1, mask_to_2], edge_index[0, mask_from_2]]).unique()

    # Combine all nodes (central + 1-hop + 2-hop)
    all_neighbors = torch.cat([neighbors_1, neighbors_2]).unique()
    nodes_in_subgraph = torch.cat([torch.tensor([node_idx]), all_neighbors]).numpy().astype(int)

    # --- 2. Build NetworkX subgraph ---
    G = nx.Graph()
    G.add_nodes_from(nodes_in_subgraph)

    nodes_tensor = torch.from_numpy(nodes_in_subgraph).to(edge_index.device)
    sub_edge_mask = (torch.isin(edge_index[0], nodes_tensor) &
                     torch.isin(edge_index[1], nodes_tensor))
    sub_edges = edge_index[:, sub_edge_mask].t().numpy()
    G.add_edges_from(sub_edges)

    # --- 3. Prepare colors and numeric labels ---
    mask_map   = {0: 'lightgray', 1: 'blue', 2: 'orange', 3: 'red'}
    mask_names = {0: 'Pool', 1: 'Train', 2: 'Val', 3: 'Test'}

    # Use numeric labels directly from graph_data.y
    labels = graph_data.y.cpu().numpy()
    label_map = {n: int(labels[n]) for n in nodes_in_subgraph}

    # Color by mask, highlight central node
    color_map = ['lime' if n == node_idx else mask_map.get(all_masks[n], 'black') for n in G.nodes()]

    # --- 4. Draw the graph ---
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1.0 / np.sqrt(max(len(G.nodes()), 1)), iterations=50, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=400, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_labels(G, pos, labels=label_map, font_size=8)

    plt.title(f"2-Hop Neighborhood of Node {node_idx} (Iter {iteration}, Seed {seed}, {dataset_name})")
    plt.axis('off')

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Selected', markerfacecolor='lime', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Pool', markerfacecolor='lightgray', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Val', markerfacecolor='orange', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10),
    ]
    plt.legend(handles=handles, title="Node Type", loc='best')

    os.makedirs('plots', exist_ok=True)
    out_path = f'plots/neighbors/neighborhood2hop_node={node_idx}_iter={iteration}_seed={seed}.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"2-hop neighborhood plot saved to '{out_path}'.")