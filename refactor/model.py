class SimpleNN():
    
    def __init__(self, input_size, hidden_size, output_size):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.criterion = nn.CrossEntropyLoss()

    def backward(self, x, y):
        x = x.reshape(1, -1)
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        self.model.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        return loss.grad.numpy()

    def forward(self, x):
        x = x.reshape(1, -1)
        x = torch.tensor(x, dtype=torch.float32)
        output = self.model(x)
        return output.detach().numpy
    
    def gradient(self, x, y):
        x = x.reshape(1, -1)
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        self.model.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        return x.grad.numpy()
