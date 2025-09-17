# model and datasets factory
from models import *
from preprocess_data import *

model_factory = {
        "resnet18": ResNet18(), #TODO: manually change the resnet last layer to 10 output classes
        "gcn": GraphSAGE(hidden_channels=10, output_dim=10, seed=322228),
        "fraudnet": CreditCardFraudNet().double()
        # add every model that gets added here
}

dataset_factory = {
    "cifar10": get_cifar10_data("./data/cifar10"),
    "credit-cards": get_credit_data("./data/credircard"),
    "IMDB": get_imdb_data("./data/IMDB.csv")
    # add every path to the dataset that gets added here

}

criterias_factory = {
    "random": 1488322
}


def load_model_wrapper(model_name, n_classes):
    if model_name not in model_factory:
        raise ValueError(f"Unknown model {model_name}")
    return model_factory[model_name]

def load_dataset_wrapper(dataset_name):
    if dataset_name not in dataset_factory:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_factory[dataset_name]

            