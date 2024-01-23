import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import util
import time
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 100, 3)
        self.conv3 = nn.Conv2d(100, 200, 3)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.to(device)
epochs = 2
batch_size = 64
train_arr, train_label, val_arr, val_label = util.dataToArr("./Data/train.csv", has_label = 1, batch_size = batch_size)
train_arr.astype(np.float32)
val_arr.astype(np.float32)
