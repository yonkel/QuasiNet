import pickle
from datetime import time

import torch
from utils.Nets import CifarNet

from utils.data import get_bin_cifar_dataset
from utils.trainer_convergence import test_one_setup

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
