#import pandas as pd

#df0 = pd.read_csv('refactor\mnist_train.csv')
#df1 = pd.read_csv('refactor\mnist_test.csv')
#columns = ['number_label'] + [f'pixel_{i}' for i in range(784)]
#df0.columns = columns
#df1.columns = columns
#df0.to_pickle('mnist_train.pkl')
#df1.to_pickle('mnist_test.pkl')
#print(df0)
#print(df1)

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from model import SimpleNN

# Load the data
with open('mnist_train.pkl', 'rb') as fid:
    mnist_train = pickle.load(fid)

# Preprocess the data
mnist_mean = 0.1307
mnist_std = 0.3081
X_train = mnist_train.iloc[:100, 1:].values.astype(np.float32) / 255.0
X_train = (X_train - mnist_mean) / mnist_std
y_train = mnist_train.iloc[:100, 0].values.astype(np.int64)

# Create DataLoader
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize and train the model
input_size = 784
hidden_size = 128
output_size = 10
model = SimpleNN(input_size, hidden_size, output_size)
model.train(train_loader, epochs=10)

# Save the trained model
#torch.save(model.state_dict(), 'model.pth')
with open('model.pkl', 'wb') as fid:
    pickle.dump(model, fid)