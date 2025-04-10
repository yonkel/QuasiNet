import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from modules.quasi import Quasi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps' for macos


def train_one_net(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_set: torch.utils.data.Dataset, # we use set because it is easier to get the size than with dataloader
        batch_size: int,
        max_epochs: int,
        zero_label: int,
        criterion: torch.nn.modules.loss = torch.nn.MSELoss(),
        eval_set=None,
    ):

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
            outputs = model(inputs).squeeze()  # TODO throws warning with batchsize 1, outputs = Size([]), labels =Size([1])

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

            print(eval_loss / len(eval_set))
            model.train()

    return False, max_epochs, train_loss_list, train_acc_list


def test_one_setup(
        net_class,
        train_set: torch.utils.data.Dataset,
        hidden: int,
        lr: float,
        repeats: int,
        batch_size: int,
        max_epochs: int,
        zero_label: int,
        criterion: torch.nn.modules.loss,
        verbose: bool = False,
):
    converged_list = []
    end_epoch_list = []
    for repeat in range(repeats):
        model = net_class()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        converged, epoch, train_loss_list, train_acc_list = train_one_net(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            # eval_set=eval_set,
            zero_label=zero_label,
            max_epochs=max_epochs,
            batch_size=batch_size,
            criterion=criterion,
        )

        if verbose:
            print(repeat, converged)

        converged_list.append(converged)
        end_epoch_list.append(epoch)

    converged_epochs = [end_epoch_list[i] for i in range(len(end_epoch_list)) if converged_list[i] == True]
    if converged_epochs == []:
        converged_epochs = [max_epochs]
    print(f"Converged {sum(converged_list)}/{repeats}, avg epoch {np.mean(converged_epochs)}, std epoch = {np.std(converged_epochs)}")


    return {
        "hidden": hidden,
        "converged": sum(converged_list),
        "lr": lr,
        "batch_size": batch_size,
        "epochs": end_epoch_list,
    }


if __name__ == '__main__':
    ...
