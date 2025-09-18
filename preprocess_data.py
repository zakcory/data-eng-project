# Here we will take a csv/any dataset and transform it into X (set of datapoints) and y (set of labels for each point)
# Then, we will pass it to the pipeline in this form
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from thinc.util import to_categorical
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd

# CIFAR10 loader
def get_cifar10_data(data_dir="./data/cifar10"):

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    # we need the entire dataset for tensor indexing later on 
    X_train = torch.from_numpy(train_set.data)
    X = torch.cat([X_train, torch.from_numpy(test_set.data)], dim=0)  
    y = torch.tensor(train_set.targets + test_set.targets, dtype=torch.long)

    # normalizing the tensor
    X = X.permute(0, 3, 1, 2).float().div_(255.0)
    mean = X.mean(dim=(0, 2, 3), keepdim=True)              
    std  = X.std(dim=(0, 2, 3), keepdim=True)  
    print(mean, std)
    X = (X - mean) / std

    return X, y


def get_glass_data(data_dir="./data/glass/glass.csv"):
    """
    Loads and preprocesses the entire glass dataset.
    """
    df = pd.read_csv(data_dir)

    # 1. Separate features (X) and the target label (y)
    X = df.drop('Type', axis=1).values
    y = df['Type'].values

    # 2. Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_tensor = torch.from_numpy(X_scaled).float()
    y_tensor = torch.from_numpy(y_encoded).long()

    return X_tensor, y_tensor




def preprocess_imdb_text(text):
    """Cleans a single text string."""
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words


def get_imdb_data(data_dir="./data/IMDB/IMDB Dataset.csv"):
    """
    Loads, cleans, and vectorizes the IMDB dataset.
    """
    print("Loading and preprocessing IMDB data... This may take a moment.")
    df = pd.read_csv(data_dir)

    # 1. Clean all reviews and create a list of word lists
    all_reviews = [preprocess_imdb_text(review) for review in df['review']]

    # 2. Build the vocabulary
    word_counts = Counter(word for review in all_reviews for word in review)
    # Sort by frequency and create word-to-integer mapping
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: i + 1 for i, word in enumerate(sorted_vocab)}  # +1 for padding token 0

    # 3. Encode the reviews into integer sequences
    reviews_encoded = []
    for review in all_reviews:
        reviews_encoded.append([word2idx.get(word, 0) for word in review])

    # 4. Pad all sequences to a fixed length (e.g., 200)
    seq_length = 200
    padded_features = np.zeros((len(reviews_encoded), seq_length), dtype=int)
    for i, row in enumerate(reviews_encoded):
        padded_features[i, -len(row):] = np.array(row)[:seq_length]

    # 5. Encode labels
    labels = np.array([1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']])

    # 6. Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(padded_features).long()
    y_tensor = torch.from_numpy(labels).long()

    # The model will need to know the size of the vocabulary
    vocab_size = len(word2idx) + 1  # +1 for the 0 padding

    print("IMDB data preprocessing complete.")
    # Return data and metadata needed for the model
    return X_tensor, y_tensor, {'vocab_size': vocab_size}