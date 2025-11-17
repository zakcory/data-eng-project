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


def build_knn_graph(embeddings, k=10, symmetrize=True) -> torch.LongTensor:
    """
    Build a k-NN graph from embeddings.
    By default, makes the graph undirected (symmetrize=True)

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

def plot_gnn_neighborhood(graph_data, all_masks, node_idx, iteration, seed, dataset_name):
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
    out_path = f'plots/neighbors/{dataset_name}/neighborhood2hop_node={node_idx}_iter={iteration}_seed={seed}.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"2-hop neighborhood plot saved to '{out_path}'.")


def build_set_inclusion_mask(self, graph_data):
        """
        Builds the mask for the purpose of plotting the kNN 2-hop neighborhood
        graph for a node

        Pool -> 0
        Train -> 1
        Val -> 2
        Test -> 3
        """
        all_masks = np.zeros(self.total_size, dtype=int)
        all_masks[graph_data.train_mask.cpu().numpy()] = 1
        all_masks[graph_data.valid_mask.cpu().numpy()] = 2
        all_masks[graph_data.test_mask.cpu().numpy()]  = 3
        return all_masks
        

def plot_tsne_embeddings(embeddings, labels, train_indices, iteration, dataset_name, criterion, seed):
    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    train_mask = np.zeros(len(labels_np), dtype=bool)
    train_mask[np.array(train_indices)] = True

    # compute tSNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='random',
        random_state=0
    )
    emb_2d = tsne.fit_transform(emb_np)
    plt.figure(figsize=(8, 6))

    # scatter by label
    plt.scatter(
        emb_2d[~train_mask, 0],
        emb_2d[~train_mask, 1],
        c=labels_np[~train_mask],
        cmap='tab10',
        alpha=0.6,
        s=10,
    )

    # highlight training nodes
    plt.scatter(
        emb_2d[train_mask, 0],
        emb_2d[train_mask, 1],
        c='black',
        s=25,
        alpha=0.9,
        label='selected nodes'
    )

    plt.title(f"{dataset_name} t-SNE, Crit: {criterion}, Iteration: {iteration}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/full_maps/{criterion}/seed={seed}_data={dataset_name}_iter={iteration}.png", dpi=300)
    plt.close()


def edge_homophily(edge_index, y):
    """
    Calculates edge-homophily score for the graph
    """
    src, dst = edge_index
    return (y[src] == y[dst]).float().mean().item()


def node_homophily(edge_index, y):
    """
    Calculates node-homophily score for the graph
    """
    src, dst = edge_index
    N = y.size(0)

    # accumulate
    same = torch.zeros(N)
    deg = torch.zeros(N)

    for u, v in zip(src, dst):
        same[u] += (y[u] == y[v]).float()
        deg[u] += 1
    
    # avoid divide-by-zero
    mask = deg > 0
    return (same[mask] / deg[mask]).mean().item()


def plot_homophily(edge_hom_margin, edge_hom_gnn, 
                   node_hom_margin, node_hom_gnn, 
                   data_name, seed):
    """
    Plot edge and node homophily across iterations
    """
    out_dir = f'plots/homo_plots'
    os.makedirs(out_dir, exist_ok=True)

    iters = list(range(1, len(edge_hom_margin) + 1))

    # node homophily
    plt.figure()
    plt.plot(iters, node_hom_margin, label='Node Homophily (pure margin)', marker='o')
    plt.plot(iters, node_hom_gnn, label='Node Homophily (gnn margin)', marker='o')

    plt.xticks(iters)
    plt.xlim(1, len(iters))
    plt.xlabel('Iteration')
    plt.ylabel('Node Homophily')
    plt.title(f'Node Homophily vs Iteration\nDataset: {data_name}, Seed: {seed}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/seed={seed}_data={data_name}_node_homophily.png')
    plt.close()

    # edge homophily
    plt.figure()
    plt.plot(iters, edge_hom_margin, label='Edge Homophily (margin)', marker='o')
    plt.plot(iters, edge_hom_gnn, label='Edge Homophily (gnn)', marker='o')

    plt.xticks(iters)
    plt.xlim(1, len(iters))
    plt.xlabel('Iteration')
    plt.ylabel('Edge Homophily')
    plt.title(f'Edge Homophily vs Iteration\nDataset: {data_name}, Seed: {seed}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/seed={seed}_data={data_name}_edge_homophily.png')
    plt.close()


def plot_uncertainty_side_by_side(
    uncert_margin_by_iter,
    uncert_gnn_by_iter,
    dataset,
    seed,
    snapshot_step=5
):
    """
    For iterations 1,5,10,15,... plot side-by-side histograms of uncertainties (pure margin vs. GNN margin)
    """
    os.makedirs("plots", exist_ok=True)
    snapshot_iters = [i for i in range(20) if (i + 1) % snapshot_step == 0 or i == 0]

    for i in snapshot_iters:
        um = uncert_margin_by_iter[i]
        ug = uncert_gnn_by_iter[i]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        # Margin side
        axes[0].hist(um, bins=40, density=True, alpha=0.8)
        axes[0].set_title(f"Pure Margin, iter {i+1}")
        axes[0].set_xlabel("Uncertainty score")
        axes[0].set_ylabel("Density")
        axes[0].grid(True, alpha=0.3)

        # GNN side
        axes[1].hist(ug, bins=40, density=True, alpha=0.8)
        axes[1].set_title(f"GNN Margin, iter {i+1}")
        axes[1].set_xlabel("Uncertainty score")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Uncertainty distributions @ iteration {i+1}\n"
            f"Dataset = {dataset}, Seed = {seed}"
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"plots/uncert_side_by_side/data={dataset}_seed={seed}_iter={i+1}.png"
        plt.savefig(fname)
        plt.close(fig)