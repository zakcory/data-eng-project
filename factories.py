# model and datasets factory
from models import *
from preprocess_data import *

model_factory = {
        "resnet18": ResNet18(n_classes=10),
        "gcn": GraphSAGE()
        # add every model that gets added here
}

dataset_factory = {
    "cifar10": get_cifar10_data("./data/cifar10"),
    "credit-cards": get_credit_data("./data/credircard.csv"),
    "IMDB": get_imdb_data("./data/IMDB.csv")
    # add every path to the dataset that gets added here

}

criterias_factory = {
    "random": 
}


def load_model_wrapper(model_name, n_classes):
    if model_name not in model_factory:
        raise ValueError(f"Unknown model {model_name}")
    return model_factory[model_name]

def load_dataset_wrapper(dataset_name):
    if dataset_name not in dataset_factory:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_factory[dataset_name]

            