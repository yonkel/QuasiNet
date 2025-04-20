import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.Quasi import Quasi


## THIS FILE IS FOR SAVING NET CLASSES FOR TORCH LOADING

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
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

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 20)
        # self.q = Quasi(20, 10)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # x = self.q(x)
        x = self.fc3(x)
        return x

class TitanicNet(nn.Module):
    def __init__(self):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(10, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class TitanicQTNet(nn.Module):
    def __init__(self):
        super(TitanicQTNet, self).__init__()
        self.q1 = Quasi(9, 16)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.q1(x)
        x = F.tanh(self.fc1(x))
        return x

class TitanicTQNet(nn.Module):
    def __init__(self):
        super(TitanicTQNet, self).__init__()
        self.fc1 = nn.Linear(9, 2)
        self.q1 = Quasi(2, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        return x

class TitanicQTQNet(nn.Module):
    def __init__(self):
        super(TitanicQTQNet, self).__init__()
        self.q1 = Quasi(9, 16)
        self.fc1 = nn.Linear(16, 8)
        self.q2 = Quasi(8, 1)


    def forward(self, x):
        x = self.q1(x)
        x = F.tanh(self.fc1(x))
        x = self.q2(x)

        return x
()

####### TESTING NETS #######

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