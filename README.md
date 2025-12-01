
# GNN-based Active Learning Framework

**Data Lab 2 - 0940295**  
**Technion - Israel Institute of Technology**  
**Team Members**: Zakhar Manikhas, Mikhail Gruntov, Dan Amler

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

### Quick Start
```bash
# Clone repository
git clone https://github.com/zakcory/data-eng-project
cd data-eng-project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for IMDB)
python -c "import nltk; nltk.download('stopwords')"

# To run a default experiment
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
```

## 🚀 Running Experiments

### Running manually crafted expiremnents

You can run the pipeline for any dataset–model pair by specifying the dataset, model, and any relevant training options.

```bash
python pipeline.py \
    --dataset_name <DATASET> \
    --model_name <MODEL> \
    [OPTIONS]
```

Supported datasets: cifar10, drybean, IMDB

Supported models:   resnet18 (CNN), beannet (MLP), lstm (LSTM)

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--iterations` | Number of AL rounds | 20 |
| `--budget_per_iter_ratio` | Fraction of data labeled per round | 0.01 |
| `--dataset_name` | Dataset to use (cifar10/IMDB/drybean) | IMDB |
| `--model_name` | Base model architecture | lstm |
| `--fine_tune` | Enable fine-tuning | False |

### Running expirements using ready-to-go script

The repository includes a convenience script (`run_all_expirements.sh`) that launches all fine-tuning experiments with predefined settings.  
It automatically activates the environment, configures deterministic CUDA options, and runs the pipeline for DryBean, IMDB, and CIFAR-10.  
Simply execute the script to reproduce all fine-tuning experiments in one step.

## 📊 Evaluation Metrics

The framework provides comprehensive analysis through:
- **Accuracy curves**: Performance over AL iterations
- **Homophily analysis**: Graph structure quality metrics
- **t-SNE visualizations**: Embedding space evolution
- **Uncertainty distributions**: Comparison of selection strategies

After running the expirements, all of the above will be under the `plots` directory
