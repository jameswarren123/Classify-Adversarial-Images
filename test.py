import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from model import SimpleNN
try:
    df0 = pd.read_csv('mnist_train.csv')
    df1 = pd.read_csv('mnist_test.csv')
    columns = ['number_label'] + [f'pixel_{i}' for i in range(784)]
    df0.columns = columns
    df1.columns = columns
    df0.to_pickle('mnist_train.pkl')
    df1.to_pickle('mnist_test.pkl')
    print(df1)
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")

try:
    with open('mnist_train.pkl', 'rb') as fid:
        mnist_train = pickle.load(fid)
except FileNotFoundError:
    print("mnist_train.pkl not found. Please ensure the file exists.")
    exit(1)

mnist_mean = 0.1307
mnist_std = 0.3081

if mnist_train.empty or len(mnist_train.columns) != 785:
    raise ValueError("Invalid MNIST data format. Expected 785 columns (label + 784 pixels)")

X_train = mnist_train.iloc[:, 1:].values.astype(np.float32) / 255.0
X_train = (X_train - mnist_mean) / mnist_std
y_train = mnist_train.iloc[:, 0].values.astype(np.int64)

if X_train.shape[1] != 784:
    raise ValueError(f"Expected 784 features, got {X_train.shape[1]}")

try:
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
except Exception as e:
    print(f"Error creating DataLoader: {e}")
    exit(1)

input_size = 784
hidden_size = 128
output_size = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(input_size, hidden_size, output_size).to(device)

def train_with_device(self, train_loader, epochs=10):
    self.model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

model.train(train_loader, epochs=10)

try:
    with open('model.pkl', 'wb') as fid:
        pickle.dump(model, fid)
except Exception as e:
    print(f"Error saving model: {e}")