"""
Data preprocessing module for Active Learning pipeline.
Handles loading and preprocessing of CIFAR-10, DryBean, and IMDB datasets.
"""

import re
import pandas as pd
import numpy as np
import torch

from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import datasets, transforms
from collections import Counter


def get_cifar10_data(data_dir="./data/cifar10"):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        data_dir: Path to CIFAR-10 data directory
    
    Returns:
        X: Image tensors [N, 3, 32, 32], normalized
        y: Label tensor [N]
        metadata: Empty dict for consistency
    """
    # Data augmentation for training (will be applied during AL)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    # Combine train and test for AL pool (will be split differently)
    X_train = torch.from_numpy(train_set.data)
    X = torch.cat([X_train, torch.from_numpy(test_set.data)], dim=0)
    y = torch.tensor(train_set.targets + test_set.targets, dtype=torch.long)

    # Normalize: convert to [0,1] then standardize
    X = X.permute(0, 3, 1, 2).float().div_(255.0)
    mean = X.mean(dim=(0, 2, 3), keepdim=True)
    std  = X.std(dim=(0, 2, 3), keepdim=True)
    X = (X - mean) / std

    return X, y, {}


def get_drybean_data(data_dir="./data/Dry_Bean.csv"):
    """
    Load and preprocess Dry Bean tabular dataset.
    
    Args:
        data_dir: Path to CSV file
    
    Returns:
        X: Feature tensor [N, 16], standardized
        y: Label tensor [N] with encoded classes
        metadata: Empty dict for consistency
    """
    df = pd.read_csv(data_dir)

    # Separate features (X) and the target label (y)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode the string labels into integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(X_scaled).float()
    y_tensor = torch.from_numpy(y_encoded).long()

    # The dataset loader should return a metadata object for consistency, even if it's empty
    return X_tensor, y_tensor, {}


def preprocess_imdb_text(text):
    """
    Clean and tokenize a single movie review.
    
    Args:
        text: Raw review text
    
    Returns:
        List of cleaned word tokens
    """

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
    Load and preprocess IMDB movie review dataset for sentiment analysis.
    
    Creates vocabulary, encodes reviews as integer sequences, and pads to fixed length.
    
    Args:
        data_dir: Path to IMDB CSV file
    
    Returns:
        X: Padded sequence tensor [N, seq_length]
        y: Binary sentiment labels [N] (0=negative, 1=positive)
        metadata: Dict containing vocab_size for LSTM model
    """

    print("Loading and preprocessing IMDB data... This may take a moment.")
    df = pd.read_csv(data_dir)

    # Clean all reviews
    all_reviews = [preprocess_imdb_text(review) for review in df['review']]

    # Build vocabulary from word frequencies
    word_counts = Counter(word for review in all_reviews for word in review)

    # Create word-to-index mapping
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: i + 1 for i, word in enumerate(sorted_vocab)}  # +1 for padding token 0

    # Encode reviews as integer sequences
    reviews_encoded = []
    for review in all_reviews:
        reviews_encoded.append([word2idx.get(word, 0) for word in review])

    # Pad sequences to fixed length
    seq_length = 200  # Max sequence length
    padded_features = np.zeros((len(reviews_encoded), seq_length), dtype=int)
    for i, row in enumerate(reviews_encoded):
        # Right-pad: place sequence at the end
        padded_features[i, -len(row):] = np.array(row)[:seq_length]

    # Encode labels
    labels = np.array([1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']])

    # Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(padded_features).long()
    y_tensor = torch.from_numpy(labels).long()

    # Store vocabulary size for model initialization
    vocab_size = len(word2idx) + 1  # +1 for the 0 padding

    print("IMDB data preprocessing complete.")
    
    return X_tensor, y_tensor, {'vocab_size': vocab_size}