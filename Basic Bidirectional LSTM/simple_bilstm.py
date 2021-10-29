# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:58:07 2021

@author: TOYGARTANYEL
"""

# Imports 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Creating BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out

# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters/optimization
input_size = 28
sequence_length = 28
num_layers = 2 
hidden_size = 128
num_classes= 10
learning_rate = 0.01
batch_size = 32
num_epochs = 2

# Loading the data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initializing network
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        #opening cuda if possible
        data = data.to(device = device).squeeze(1)
        targets = targets.to(device = device)
        
        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        #basic backward
        optimizer.zero_grad()
        loss.backward()
        
        #gradient descent or adam
        optimizer.step()



# Checking accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy of training data")
    else: 
        print("checking accuracy of test data")
    
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device).squeeze(1)
            y = y.to(device = device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'{num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)