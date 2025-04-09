import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.feedforward_nn import train_feedforward_model

def test_train_feedforward_model():
    # Create small dummy dataset
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, 100)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model, train_losses, val_losses, train_acc, val_acc = train_feedforward_model(loader, loader, input_size=20, num_classes=2, num_epochs=1)
    # Check that the losses lists have one entry per epoch (here, 1)
    assert len(train_losses) == 1
    assert len(val_losses) == 1
