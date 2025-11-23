"""
Factory module for model and dataset loading in the GNN pipeline.
Provides centralized registration and instantiation of models and datasets.
"""

from models import *
from preprocess_data import get_cifar10_data, get_drybean_data, get_imdb_data

# Model Registry
# Maps string identifiers to model class constructors
model_factory = {
    "resnet18": ResNet18,      # CNN for image classification (CIFAR-10)
    "gcn": GraphSAGE,          # Graph Neural Network for label propagation
    "beannet": BeanNet,        # MLP for tabular data (DryBean dataset)
    "lstm": SentimentLSTM      # LSTM for text classification (IMDB)
}

# Dataset Registry  
# Maps string identifiers to dataset loader functions (not loaded data)
dataset_factory = {
    "cifar10": get_cifar10_data,   # Computer vision dataset: 32x32 RGB images, 10 classes
    "drybean": get_drybean_data,   # Tabular dataset: 16 features, 7 bean types
    "IMDB": get_imdb_data          # NLP dataset: Movie reviews, binary sentiment
}


def load_model_wrapper(model_name, n_classes, model_config, input_dim=None):
    """
    Instantiate a model with appropriate configuration. 
    
    Args:
        model_name: Model identifier from model_factory
        n_classes: Number of output classes
        model_config: Dict of model-specific configs (required for LSTM)
        input_dim: Input dimension (required for BeanNet)
    Returns:
        Instantiated model ready for training
    """
    if model_name not in model_factory:
        raise ValueError(f"Unknown model {model_name}")

    ModelClass = model_factory[model_name]  # Get the class

    if model_name == "lstm":
        if model_config is None:
            raise ValueError("model_config must be provided for LSTM")
        return ModelClass(**model_config)

    elif model_name == "resnet18":
        return ModelClass(num_classes=n_classes)

    elif model_name == "beannet":
        if input_dim is None:
            raise ValueError("input_dim must be provided for BeanNet")
        return ModelClass(input_dim=input_dim, num_classes=n_classes)
    else:
        return ModelClass()


def load_dataset_wrapper(dataset_name):
    """
    Load and preprocess a dataset.

    Args:
        dataset_name: Dataset identifier from dataset_factory
    Returns:
        Preprocessed dataset components
    """
    if dataset_name not in dataset_factory:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Dataset file paths
    paths = {
        "cifar10": "./data/cifar10",
        "drybean": "./data/Dry_Bean.csv", 
        "IMDB": "./data/IMDB/IMDB Dataset.csv"
    }

    data_loader_func = dataset_factory[dataset_name]  # Get the function
    # Call it with the path, and handle the possibility of it returning three values
    result = data_loader_func(paths[dataset_name])
    return result