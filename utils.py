import matplotlib.pyplot as plt
import torch
import numpy as np


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
