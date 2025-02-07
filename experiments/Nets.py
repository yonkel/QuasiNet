import torch.nn as nn
import torch.nn.functional as F
import torch


## THIS FILE IS FOR SAVING NET CLASSES FOR TORCH LOADING

class Net_test(nn.Module):
    def __init__(self):
        super(Net_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Plus(nn.Module):
    def __init__(self):
        super(Plus, self).__init__()

    def forward(self, x):
        return x + torch.ones(x.size()).type_as(x)

class Minus(nn.Module):
    def __init__(self):
        super(Minus, self).__init__()

    def forward(self, x):
        return x - torch.ones(x.size()).type_as(x)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()

        self.fc1 = Plus()

    def forward(self, x):
        return self.fc1(x)