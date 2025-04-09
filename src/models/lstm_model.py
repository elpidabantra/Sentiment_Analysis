import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix):
        """Define an LSTM model using pretrained embeddings."""
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(self.relu(self.fc(x[:, -1, :])))
        return self.out(x)

def train_lstm_model(train_loader, val_loader, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix, num_epochs=5, lr=0.001):
    """Train the LSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix).to(device)
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
