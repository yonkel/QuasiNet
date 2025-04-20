import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Quasi import Quasi
from utils.data import get_parity_dataset
from utils.trainer_convergence import test_one_setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps' for macos


def test_parity(parity_degree: int,
                hidden_layers: list,
                repeats: int = 100,
                learning_rate: float = 0.5,
                batch_size: int = 1,
                zero_label: int = 0,
                max_epochs: int = 1000,
                criterion: torch.nn.modules.loss = torch.nn.MSELoss(),
                verbose: bool = False,
                ):

    class ParityNet(torch.nn.Module):
        def __init__(self):
            super(ParityNet, self).__init__()
            self.fc1 = nn.Linear(parity_degree, h)
            self.q1 = Quasi(h, 1)

        def forward(self, x):
            x = F.tanh(self.fc1(x))
            x = self.q1(x)
            return x

    train_set = get_parity_dataset(parity_degree, remap=True)
    output_list = []
    for h in hidden_layers:
        output_list.append(test_one_setup(net_class=ParityNet,
                                          train_set=train_set,
                                          hidden=h,
                                          lr=learning_rate,
                                          repeats=repeats,
                                          batch_size=batch_size,
                                          max_epochs=max_epochs,
                                          zero_label=zero_label,
                                          criterion=criterion,
                                          verbose=verbose,
                                          )
                           )
    return output_list


if __name__ == '__main__':
    results = test_parity(parity_degree=3,
                          hidden_layers=[2, 3, 4, 5],
                          repeats=10,
                          learning_rate=0.5,
                          batch_size=1,
                          zero_label=-1,
                          max_epochs=1000,
                          verbose=False
                          )
