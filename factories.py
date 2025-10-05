# File: factories.py

from models import * # Assuming you have this
from preprocess_data import get_cifar10_data, get_drybean_data, get_imdb_data

# Store the classes themselves, not instantiated objects
model_factory = {
    "resnet18": ResNet18,
    "gcn": GraphSAGE,
    "beannet": BeanNet, # Changed from glassnet
    "lstm": SentimentLSTM
}

# Store the functions themselves, not the data they return
dataset_factory = {
    "cifar10": get_cifar10_data,
    "drybean": get_drybean_data, # Changed from glass
    "IMDB": get_imdb_data
}


# The model wrapper is mostly correct, just needs the class from the factory
def load_model_wrapper(model_name, n_classes, model_config, input_dim=None):
    if model_name not in model_factory:
        raise ValueError(f"Unknown model {model_name}")

    ModelClass = model_factory[model_name]  # Get the class

    if model_name == "lstm":
        if model_config is None:
            raise ValueError("model_config must be provided for LSTM")
        return ModelClass(**model_config)

    elif model_name == "resnet18":
        return ModelClass(num_classes=n_classes)

    elif model_name == "beannet": # Changed from glassnet
        if input_dim is None:
            raise ValueError("input_dim must be provided for BeanNet")
        return ModelClass(input_dim=input_dim, num_classes=n_classes)
    else:
        return ModelClass()


# The dataset wrapper now calls the function with the correct path
def load_dataset_wrapper(dataset_name):
    if dataset_name not in dataset_factory:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Define file paths here
    paths = {
        "cifar10": "./data/cifar10",
        "drybean": "./data/Dry_Bean.csv", # Changed from glass
        "IMDB": "./data/IMDB/IMDB Dataset.csv"
    }

    data_loader_func = dataset_factory[dataset_name]  # Get the function
    # Call it with the path, and handle the possibility of it returning three values
    result = data_loader_func(paths[dataset_name])
    return result