import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter


class Quasi_selflearn(nn.Module):

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features =  in_features
        self.out_features =  out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, input: Tensor) -> Tensor: # def activation(self, input: Tensor) -> Tensor:
        # print(input)

        # batch = input.shape[0]
        while len(input.shape) < 3: # --> so the tensor shape is consistent (batch, out_features, in_features)
          input = input.unsqueeze(-2)
          # input = input.view(batch, 1, self.in_features)

        h = 1 - torch.sigmoid(self.weight) * ( 1 - input )
        # print(h.shape)
        h_prod = torch.prod(h, axis=2, keepdim=True) # axis=1 if tensor has only 2 dim

        return h_prod.squeeze(-1)

    def learning(self, act, error, learning_rate):
        error_new, w_change_new = self.delta_quasi(act, error)
        self.W[i] += learning_rate * w_change_new
        return error_new

    def delta_quasi(self, act, error):
        quasi_weights = self.sigmoid.apply_func(self.W[i])

        qW_act_mul = 1 - quasi_weights * (1 - act).T   # multiplication of compressed weights and activations
        common_term = (torch.prod(torch.where(qW_act_mul == 0, 1, qW_act_mul), axis=1, keepdims=True) / qW_act_mul).T

        error_new = torch.sum(error * common_term * quasi_weights, axis=1)

        w_change_new = error * common_term.T * (act.T - 1) * quasi_weights * (1 - quasi_weights)

        return error_new, w_change_new