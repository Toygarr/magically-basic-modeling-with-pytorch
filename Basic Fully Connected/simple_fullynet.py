# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:20:37 2021

@author: TOYGARTANYEL
"""

# Imports 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Creating Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,  128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters/optimization
input_size = 784
num_classes= 10
learning_rate = 0.01
batch_size = 128
num_epochs = 1

# Loading the data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initializing network
model = NN(input_size = input_size, num_classes = num_classes).to(device)


# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        #opening cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        #reshape for getting correct shape
        data = data.reshape(data.shape[0], -1)
        
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
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'{num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

