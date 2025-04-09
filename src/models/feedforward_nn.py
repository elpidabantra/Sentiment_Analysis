import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, num_classes):
        """Define a simple feedforward neural network."""
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_feedforward_model(train_loader, val_loader, input_size, num_classes, num_epochs=5, lr=0.0005):
    """Train the Feedforward NN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedforwardNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(correct / total)
    
        # Validation
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = total_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(correct / total)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss={avg_train_loss:.4f} | Train Acc={train_accuracies[-1]:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_accuracies[-1]:.4f}")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies
