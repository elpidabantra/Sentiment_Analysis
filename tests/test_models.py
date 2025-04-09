import torch
import pytest
from src.models.feedforward_nn import FeedforwardNN
from src.models.lstm_model import LSTMModel
import numpy as np

def test_feedforward_nn_forward():
    model = FeedforwardNN(input_size=10, num_classes=3)
    dummy_input = torch.randn((5, 10))
    output = model(dummy_input)
    # Check output shape [batch_size, num_classes]
    assert output.shape == (5, 3)

def test_lstm_model_forward():
    vocab_size = 20
    embed_dim = 50
    hidden_dim = 32
    num_classes = 3
    # Create a random embedding matrix
    embedding_matrix = np.random.rand(vocab_size, embed_dim)
    model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix)
    dummy_input = torch.randint(0, vocab_size, (5, 10))  # batch_size=5, seq_length=10
    output = model(dummy_input)
    assert output.shape == (5, num_classes)
