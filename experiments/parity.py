import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from utils import get_parity_dataset
from modules.quasi import Quasi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps' for macos

def train_parity(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_set: torch.utils.data.Dataset,
        batch_size: int,
        zero_label: int,  #TODO yeah..
        criterion: torch.nn.modules.loss = torch.nn.MSELoss(),
        max_epochs: int = 1000,
        eval_set: DataLoader = None,
    ):

    # for item in model.parameters():
    #     print(item)

    threshold = 0.5
    if zero_label == -1:
        threshold = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if eval_set:
        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    train_loss_list = []
    train_acc_list = []

    model.to(device)

    for epoch in range(max_epochs):
        running_loss = 0.0
        running_acc = 0
        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze() # TODO throws warning with batchsize 1, outputs = Size([]), labels =Size([1])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            rounded_labels = torch.where(outputs > threshold, 1, zero_label)
            running_acc += torch.sum(rounded_labels == labels)


        train_loss_list.append(running_loss / len(train_set))
        train_acc_list.append(running_acc / len(train_set))

        # if epoch % 10 == 0:
        #     print(train_loss_list[-1])
        #     print(train_acc_list[-1])

        if train_acc_list[-1] == 1:
            return True, epoch, train_loss_list, train_acc_list


        if eval_set is not None and epoch % 10 == 0:
            model.eval()

            eval_loss = 0.0
            for inputs, labels in eval_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

            print(eval_loss/ len(eval_set))
            model.train()

    return False, max_epochs, train_loss_list, train_acc_list



def test_convergence(
        parity_degree: int = 2,
        repeats: int = 100,
        hidden: int = 3,
        lr: float = 0.9,
        batch_size: int = 1,
    ):

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(parity_degree, hidden)
            self.q1 = Quasi(hidden, 1)

        def forward(self, x):
            x = F.tanh(self.fc1(x))
            x = self.q1(x)
            return x

    train_set = get_parity_dataset(parity_degree, remap=True)
    converged_sum = 0

    end_epoch_list = []
    for repeat in range(repeats):

        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        converged, epoch, train_loss_list, train_acc_list = train_parity(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            zero_label=-1,
            max_epochs=1000,
            batch_size=1
        )

        converged_sum += converged
        end_epoch_list.append(epoch)

    print(f"Converged {converged_sum}/{repeats}, avg epoch {np.mean(end_epoch_list)}, std epoch = {np.std(end_epoch_list)}")


    return {
    "parity_degree": parity_degree,
    "hidden": hidden,
    "converged" : converged_sum,
    "lr": lr,
    "batch_size": batch_size,
    "epochs": end_epoch_list,
    }


def test_parity(parity_degree:int,
                hidden_layers:list
    ):
    output_list = []
    for h in hidden_layers:
        output_list.append(test_convergence(parity_degree=parity_degree, repeats=100, hidden=h, lr=0.5))

    return output_list



if __name__ == '__main__':
    ...