"""
This package contains model definitions and related training functions.

Modules:
    feedforward_nn: Defines the FeedforwardNN model and its training function.
    lstm_model: Defines the LSTMModel model and its training function.
"""

# Optionally, you can expose functions/classes here:
from .feedforward_nn import FeedforwardNN, train_feedforward_model
from .lstm_model import LSTMModel, train_lstm_model
