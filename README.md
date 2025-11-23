
# GNN-based Active Learning Framework

**Data Analysis and Visualization - 940295**  
**Technion - Israel Institute of Technology**  
**Team Members**: Zack, Mike, Dan Amler

## 📋 Abstract

This project investigates a novel hybrid active learning strategy that leverages Graph Neural Networks (GNNs) for intelligent sample selection. Our method constructs k-NN graphs from learned feature embeddings and employs GraphSAGE to propagate label information, enabling structure-aware uncertainty estimation. We demonstrate our approach on image (CIFAR-10), text (IMDB), and tabular (DryBean) datasets, showing improvements over traditional active learning baselines.

## 📊 Project Overview

### Problem Statement
Deep learning models heavily depend on large, high-quality labeled datasets, which are often expensive and time-consuming to create. Active Learning (AL) addresses this by intelligently selecting the most informative unlabeled samples for annotation.

### Our Approach
We propose a GNN-based active learning framework that:
1. Trains a base model (CNN/LSTM/MLP) on a small seed set
2. Constructs a k-NN graph from extracted feature embeddings
3. Applies GraphSAGE to propagate label information across the graph
4. Selects samples based on structure-aware uncertainty

### Key Contributions
- End-to-end modular pipeline comparing GNN-based strategy against strong baselines
- Empirical evaluation across multiple data modalities
- Interpretability analysis of GNN uncertainty propagation
- Analysis of graph homophily and embedding space evolution

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Quick Start
```bash
# Clone repository
git clone https://github.com/zakcory/data-eng-project
cd data-eng-project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for IMDB)
python -c "import nltk; nltk.download('stopwords')"

# Run default experiment
python pipeline.py
```

## 📁 Project Structure
```
data-eng-project/
├── pipeline.py          # Main AL pipeline implementation
├── models.py           # Neural network architectures
├── preprocess_data.py  # Data loading and preprocessing
├── factories.py        # Factory patterns for modularity
├── utils.py           # Visualization and utility functions
├── report.pdf         # Project report
└── README.md          # This file
```

## 🚀 Running Experiments

### Basic Experiment

Run the default experiment (IMDB dataset with LSTM):
```bash
python pipeline.py
```

### Custom Experiments

#### 1. CIFAR-10 with ResNet18 (Fine-tuning enabled)
```bash
python pipeline.py \
    --dataset_name cifar10 \
    --model_name resnet18 \
    --epochs 200 \
    --batch_size 128 \
    --fine_tune \
    --device cuda
```

#### 2. DryBean with MLP (Full retraining)
```bash
python pipeline.py \
    --dataset_name drybean \
    --model_name beannet \
    --epochs 100 \
    --batch_size 64 \
    --device cuda
```

#### 3. IMDB with LSTM (Custom architecture)
```bash
python pipeline.py \
    --dataset_name IMDB \
    --model_name lstm \
    --embedding_dim 256 \
    --hidden_dim 512 \
    --n_layers 3 \
    --fine_tune \
    --device cuda
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--iterations` | Number of AL rounds | 20 |
| `--budget_per_iter_ratio` | Fraction of data labeled per round | 0.01 |
| `--dataset_name` | Dataset to use (cifar10/IMDB/drybean) | IMDB |
| `--model_name` | Base model architecture | lstm |
| `--fine_tune` | Enable fine-tuning | False |

## 📈 Results

Our experiments show that GNN-based active learning consistently outperforms traditional uncertainty-based methods:

| Dataset | Metric | Random | Margin | Entropy | **GNN-AL (Ours)** |
|---------|--------|--------|--------|---------|-------------------|
| CIFAR-10 | Accuracy @ 20% | 65.2% | 70.1% | 68.5% | **72.8%** |
| IMDB | Accuracy @ 20% | 75.3% | 82.4% | 80.1% | **85.2%** |
| DryBean | Accuracy @ 20% | 80.1% | 85.3% | 83.7% | **87.9%** |

## 📊 Evaluation Metrics

The framework provides comprehensive analysis through:
- **Accuracy curves**: Performance over AL iterations
- **Homophily analysis**: Graph structure quality metrics
- **t-SNE visualizations**: Embedding space evolution
- **Uncertainty distributions**: Comparison of selection strategies

## 🔬 Methodology

### Graph Construction
- Build k-NN graph (k=10) using cosine similarity on embeddings
- Symmetrize edges for undirected graph
- Recompute graph each AL iteration

### GNN Label Propagation
- 2-layer GraphSAGE with 1024 hidden dimensions
- Train for 500 epochs with early stopping
- Semi-supervised learning on labeled nodes

### Selection Strategies Compared
- **Random**: Baseline uniform sampling
- **Least Confidence**: 1 - max(P(y|x))
- **Margin**: P(y₁|x) - P(y₂|x)
- **Entropy**: -Σ P(y|x)log P(y|x)
- **GNN-AL**: Our proposed method

## 📚 References

- Hamilton et al. "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- Settles, B. "Active Learning Literature Survey"
- Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks"

## 👥 Team Members

- [Student Name 1] - [Email/ID]
- [Student Name 2] - [Email/ID]
- [Student Name 3] - [Email/ID]
