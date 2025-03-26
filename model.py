import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SimpleNN(nn.Module):
    """A simple two-layer neural network."""
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output
    
    def predict(self, x):
        with torch.no_grad():
            x = x.reshape(-1, 1, 28, 28)
            x = torch.tensor(x, dtype=torch.float32)
            output = self(x)
            return torch.softmax(output, dim=1).detach().numpy()
        
    def gradient(self, x, y):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        self.eval()
        x.requires_grad = True
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.zero_grad()
        loss.backward()
        return x.grad.numpy()