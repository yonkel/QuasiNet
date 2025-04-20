import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter


class QuasiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, parameters):
        while len(input.shape) < 3: # --> so the tensor shape is consistent (batch, out_features, in_features)
          input = input.unsqueeze(-2)

        h = 1 - torch.sigmoid(parameters) * ( 1 - input )
        h_prod = torch.prod(h, axis=2, keepdim=True) # axis=1 if tensor has only 2 dim

        ctx.save_for_backward(input, parameters, h, h_prod)

        return h_prod.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_out):
        input, parameters, h, h_prod = ctx.saved_tensors
        #s_params = torch.sigmoid(parameters).unsqueeze(0) # batch
        s_params = torch.sigmoid(parameters)
        cmt = h_prod/h

        grad_out = grad_out.unsqueeze(-1) # batch
        # grad_params = grad_out * cmt * (input - 1) * s_params * (1 - s_params)
        # grad_input = torch.sum(grad_out * cmt * s_params , axis=1).squeeze(1) # batch .squeeze
        grad_params = grad_out * cmt * (input - 1) * s_params * (1 - s_params)
        grad_input = torch.sum(grad_out * cmt * s_params, axis=1)  # batch.squeeze(1)

        return grad_input, grad_params

# Module wrapper for function
class QuasiModule(nn.Module):

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, input: Tensor) -> Tensor:
        return QuasiFunction.apply(input, self.weight)
