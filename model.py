import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNN(nn.Module):
    """A simple two-layer neural network."""
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_loss_gradient(self, x, y):
        """Compute loss gradient for a single example."""
        try:
            x = x.reshape(1, -1)
            if x.shape[1] != self.input_size:
                raise ValueError(f"Expected input size {self.input_size}, got {x.shape[1]}")
            x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            self.model.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            return loss.item()
        except Exception as e:
            print(f"Error in compute_loss_gradient: {e}")
            return None

    def predict(self, x):
        """Make predictions with softmax probabilities."""
        self.model.eval()
        with torch.no_grad():
            x = x.reshape(-1, self.input_size)
            x = torch.tensor(x, dtype=torch.float32)
            output = self.model(x)
            return torch.softmax(output, dim=1).detach().numpy()
    
    def gradient(self, x, y):
        """Compute input gradient."""
        try:
            x = x.reshape(-1, self.input_size)
            x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            self.model.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            return x.grad.numpy() if x.grad is not None else None
        except Exception as e:
            print(f"Error in gradient: {e}")
            return None
    
    def train(self, train_loader, epochs=10):
        """Train the model with a DataLoader."""
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')