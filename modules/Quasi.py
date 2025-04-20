import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter


class Quasi(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features =  in_features
        self.out_features =  out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0)

    def __repr__(self):
        return f"Quasi(in_features={self.in_features}, out_features={self.out_features}, bias=False)"

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() == 1: # --> add batch dimension if it's so the tensor shape is consistent (batch, out_features, in_features)
            input = input.unsqueeze(0)

        x = input.unsqueeze(1)

        h = 1 - torch.sigmoid(self.weight) * (1 - x)
        result = torch.prod(h, dim=2)

        return result


if __name__ == '__main__':

    q = Quasi(10, 5)
    w = torch.tensor([
        [0.7875, 0.4008, 0.6059, 0.3469, 0.7291, 0.6232, 0.7926, 0.1019, 0.3405, 0.4138],
        [0.2748, 0.9126, 0.7338, 0.9343, 0.4337, 0.0290, 0.1881, 0.4549, 0.0079, 0.7771],
        [0.2789, 0.7845, 0.3458, 0.8256, 0.5474, 0.6373, 0.3323, 0.0291, 0.1438, 0.1089],
        [0.8338, 0.7457, 0.6759, 0.7751, 0.9579, 0.6779, 0.1361, 0.9070, 0.9394, 0.1012],
        [0.8971, 0.2670, 0.3104, 0.9343, 0.7357, 0.4645, 0.6462, 0.7961, 0.9404, 0.7647]
    ], requires_grad=True)
    q.weight = nn.Parameter(w)

    x = torch.tensor([
        [0.5836, 0.5555, 0.0326, 0.3965, 0.9331, 0.2323, 0.7176, 0.3473, 0.4592, 0.4075],
        [0.2215, 0.4602, 0.8460, 0.0724, 0.7131, 0.9075, 0.0559, 0.7249, 0.9910, 0.5915]],
        requires_grad=True)
    d = q(x)

    # tensor([[0.0141, 0.0132, 0.0174, 0.0093, 0.0109],
    #         [0.0220, 0.0233, 0.0263, 0.0220, 0.0159]])
