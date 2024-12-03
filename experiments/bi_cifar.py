import pickle
from datetime import time


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from utils.utils import get_bin_cifar_dataset
from modules.quasi import Quasi
from utils.trainer import test_one_setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps' for macos

def test_cifar(
        repeats:int = 10,
        learning_rate:float = 0.5,
        batch_size: int = 1,
        zero_label: int = 0,
        max_epochs: int = 5000,
        criterion: torch.nn.modules.loss = torch.nn.MSELoss(),
        verbose: bool = False,
    ):

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

    train_set = get_bin_cifar_dataset(small=True)

    start = time.time()

    results_dict = test_one_setup(net_class=CifarNet,
                          train_set=train_set,
                          lr=learning_rate,
                          repeats=repeats,
                          batch_size=batch_size,
                          max_epochs=max_epochs,
                          zero_label=zero_label,
                          criterion=criterion,
                          verbose=verbose,
                          )

    end = time.time()

    net = CifarNet()
    results_dict["net"] = str(net)
    results_dict["time"] = end - start

    return results_dict



if __name__ == '__main__':

    results = test_cifar(
        repeats=10,
        learning_rate=0.5,
        batch_size = 1,
        zero_label = -1,
        max_epochs = 1000,
        verbose = False
    )


    filename = 'my_dict.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(results, file)
