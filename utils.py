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
        # optional: for large N you may want to move to CPU to save GPU mem
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


def generate_tsne_plot(embeddings, labels, masks, gnn_probs, iteration, seed, dataset_name, fine_tune, subsample=5000):
    """
    Generates and saves t-SNE plots for GNN visualization.
    'masks' is expected to be a numpy array where 0=Pool, 1=Train, 2=Val, 3=Test
    """
    print("Generating t-SNE plots...")

    # --- 1. Prepare Data ---

    # Subsample if the dataset is too large (t-SNE is slow)
    if len(labels) > subsample:
        N = len(labels)
        # Ensure we get a good mix of train/val nodes
        train_val_test_idx = np.where(masks > 0)[0]
        pool_idx = np.where(masks == 0)[0]

        # Keep all train/val/test nodes (if they are few) and sample the rest from the pool
        pool_sample_size = subsample - len(train_val_test_idx)
        if pool_sample_size > 0:
            pool_sub_idx = np.random.choice(pool_idx, pool_sample_size, replace=False)
            indices = np.concatenate([train_val_test_idx, pool_sub_idx])
        else:
            # If we have more than 'subsample' train/val/test nodes, just sample from them
            indices = np.random.choice(train_val_test_idx, subsample, replace=False)
    else:
        indices = np.arange(len(labels))

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=seed)

    # Detach and move embeddings to CPU
    embeddings_cpu = embeddings[indices].detach().cpu().numpy()
    tsne_results = tsne.fit_transform(embeddings_cpu)

    # Get corresponding labels and masks (already numpy)
    labels_sub = labels[indices].detach().cpu().numpy()
    mask_sub = masks[indices]

    # Calculate GNN uncertainty (margins) for all subsampled nodes
    gnn_probs_sub = gnn_probs[indices].detach().cpu()
    sorted_probs, _ = torch.sort(gnn_probs_sub, dim=1, descending=True)
    margins = (sorted_probs[:, 0] - sorted_probs[:, 1])
    uncertainty = 1.0 - margins  # Invert so high values = high uncertainty

    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'label': labels_sub,
        'mask': mask_sub,
        'uncertainty': uncertainty
    })

    # --- 2. Create Plot 1: Data Split & True Labels ---

    # Map mask names
    mask_map = {0: 'Pool', 1: 'Train', 2: 'Validation', 3: 'Test'}
    df['Data Split'] = df['mask'].map(mask_map)

    # Create the plot
    plt.figure(figsize=(12, 10))
    # Use different markers for data split
    markers = {0: 'o', 1: 'X', 2: 's', 3: 'P'}

    for mask_val, m in markers.items():
        subset = df[df['mask'] == mask_val]
        plt.scatter(
            subset['tsne_1'], subset['tsne_2'],
            c=subset['label'],
            label=mask_map[mask_val],
            marker=m,
            cmap='coolwarm',
            alpha=0.7,
            s=50 if m != 'o' else 20  # Make non-pool nodes bigger
        )

    plt.title(
        f't-SNE of Embeddings (Iteration {iteration}) - Labels & Data Split\nSeed: {seed}, Dataset: {dataset_name}, FT: {fine_tune}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Create legends
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    # Manually create a legend for colors (labels)
    color_handles = [plt.Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor=plt.cm.coolwarm(0.0),
                                markersize=10),
                     plt.Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor=plt.cm.coolwarm(1.0),
                                markersize=10)]

    legend1 = plt.legend(handles=handles, labels=labels_leg, title="Data Split", loc='upper right')
    plt.gca().add_artist(legend1)
    plt.legend(handles=color_handles, title="True Label", loc='lower right')

    # Save the plot
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/tsne_labels_iter={iteration}_seed={seed}_{dataset_name}.png')
    plt.close()

    # --- 3. Create Plot 2: GNN Uncertainty ---

    plt.figure(figsize=(12, 10))
    plt.scatter(
        df['tsne_1'], df['tsne_2'],
        c=df['uncertainty'],  # Color by GNN uncertainty
        cmap='viridis',  # Use a sequential colormap
        alpha=0.7,
        s=20
    )
    plt.title(
        f't-SNE of Embeddings (Iteration {iteration}) - GNN Uncertainty\nSeed: {seed}, Dataset: {dataset_name}, FT: {fine_tune}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='GNN Uncertainty (1 - Margin)')

    plt.savefig(f'{plot_dir}/tsne_uncertainty_iter={iteration}_seed={seed}_{dataset_name}.png')
    plt.close()
    print("t-SNE plots saved to 'plots/' directory.")


def plot_gnn_neighborhood(graph_data, all_masks, node_idx, iteration, seed, dataset_name):
    """
    Plots the 1-hop neighborhood of a specific node.
    'all_masks' is expected to be a numpy array where 0=Pool, 1=Train, 2=Val, 3=Test
    """
    print(f"Plotting neighborhood for node {node_idx}...")

    # --- 1. Find 1-hop neighborhood ---
    edge_index = graph_data.edge_index.cpu()
    node_idx = int(node_idx)  # Ensure it's a standard int

    # Find all edges connected to the target node
    mask_to = (edge_index[0] == node_idx)
    mask_from = (edge_index[1] == node_idx)

    neighbors_to = edge_index[1, mask_to]
    neighbors_from = edge_index[0, mask_from]

    all_neighbors = torch.cat([neighbors_to, neighbors_from]).unique()

    # Add the central node itself
    nodes_in_subgraph = torch.cat([torch.tensor([node_idx]), all_neighbors]).numpy().astype(int)

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes_in_subgraph)

    # Add edges *only* between nodes in our subgraph
    sub_edge_mask = (torch.isin(edge_index[0], nodes_in_subgraph) &
                     torch.isin(edge_index[1], nodes_in_subgraph))
    sub_edges = edge_index[:, sub_edge_mask].t().numpy()
    G.add_edges_from(sub_edges)

    # --- 2. Prepare labels and colors ---

    # 0=Pool, 1=Train, 2=Validation, 3=Test
    mask_map = {0: 'lightgray', 1: 'blue', 2: 'orange', 3: 'red'}
    mask_names = {0: 'Pool', 1: 'Train', 2: 'Val', 3: 'Test'}

    # Get true labels (0=Neg, 1=Pos)
    labels = graph_data.y.cpu().numpy()
    label_map = {n: f"{'Pos' if labels[n] == 1 else 'Neg'} ({mask_names.get(all_masks[n], 'Unknown')})" for n in
                 nodes_in_subgraph}

    # Assign colors based on mask
    color_map = []
    for node in G:
        if node == node_idx:
            color_map.append('lime')  # Highlight the central node
        else:
            color_map.append(mask_map.get(all_masks[node], 'black'))

    # --- 3. Draw the graph ---
    plt.figure(figsize=(15, 10))
    # Use a layout that spreads nodes out
    pos = nx.spring_layout(G, k=1.0 / np.sqrt(len(G.nodes())), iterations=50, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=400, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_labels(G, pos, labels=label_map, font_size=8)

    plt.title(f"GNN Neighborhood of Node {node_idx} (Iter {iteration}, Seed {seed}, {dataset_name})")
    plt.axis('off')  # Hide axes

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10)
               for label, color in
               [('Selected', 'lime'), ('Train', 'blue'), ('Pool', 'lightgray'), ('Val', 'orange'), ('Test', 'red')]]
    plt.legend(handles=handles, title="Node Type", loc='best')

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/neighborhood_node={node_idx}_iter={iteration}_seed={seed}.png')
    plt.close()
    print(f"Neighborhood plot saved to 'plots/'.")