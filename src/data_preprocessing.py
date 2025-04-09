import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import CSV_DATA_PATH



def load_text_data(csv_path=CSV_DATA_PATH):
    """
    Load text data from a CSV file.
    
    The CSV file is expected to have at least two columns: "text" and "label".
    
    Parameters:
        csv_path (str): Path to the CSV file. Defaults to CSV_DATA_PATH from config.
        
    Returns:
        texts (list): List of text strings.
        labels (list): List of corresponding labels.
        
    Raises:
        FileNotFoundError: If the CSV file is not found at the specified path.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Please ensure the file exists in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    # Adjust the column names if necessary.
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


def prepare_text_dataset(csv_path=CSV_DATA_PATH, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load text data from a CSV file and split it into training, validation, and test sets.
    
    Parameters:
        csv_path (str): Path to the CSV file (default is CSV_DATA_PATH from config).
        test_size (float): Fraction of data to be used for testing (default 0.2).
        val_size (float): Fraction of the remaining data (after test split) for validation (default 0.1).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train (list): Training texts.
        X_val (list): Validation texts.
        X_test (list): Test texts.
        y_train (list): Training labels.
        y_val (list): Validation labels.
        y_test (list): Test labels.
    """
    texts, labels = load_text_data(csv_path)
    # First split into training+validation and testing.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    # Now split training+validation into separate training and validation sets.
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_dummy_text_data(num_samples):
    """
    Generate dummy text data for demonstration purposes.
    
    Parameters:
        num_samples (int): Number of sample texts to generate.
        
    Returns:
        List[str]: A list containing the same sample sentence repeated.
    """
    return ["This is a sample sentence for text classification"] * num_samples


def generate_dummy_tfidf_data(num_samples, num_features, num_classes):
    """
    Generate dummy TF-IDF features and random labels.
    
    This can be used as a placeholder for testing model pipelines that expect numeric features.
    
    Parameters:
        num_samples (int): Total number of samples.
        num_features (int): Number of features (e.g., vocabulary size) for TF-IDF.
        num_classes (int): Number of classes for classification.
        
    Returns:
        X_train (np.ndarray): Training feature matrix.
        X_val (np.ndarray): Validation feature matrix.
        X_test (np.ndarray): Test feature matrix.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
        y_test (np.ndarray): Test labels.
    """
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size

    X_train = np.random.rand(train_size, num_features)
    X_val = np.random.rand(val_size, num_features)
    X_test = np.random.rand(test_size, num_features)
    y_train = np.random.randint(0, num_classes, train_size)
    y_val = np.random.randint(0, num_classes, val_size)
    y_test = np.random.randint(0, num_classes, test_size)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
