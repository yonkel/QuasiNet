import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter

from modules.Quasi import Quasi


class QuasiSkip(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.quasi = Quasi(in_features, out_features, **factory_kwargs)
        self.in_features = in_features
        self.out_features = out_features

        # Projection layer is needed when dimensions don't match
        self.need_projection = in_features != out_features
        if self.need_projection:
            self.projection = nn.Linear(in_features, out_features, bias=False, **factory_kwargs)

    def __repr__(self):
        return f"QuasiSkip(in_features={self.in_features}, out_features={self.out_features})"

    def forward(self, input: Tensor) -> Tensor:
        # Store original input for skip connection
        identity = input

        # Handle single dimension input
        if input.dim() == 1:
            identity = identity.unsqueeze(0)

        # Pass through Quasi layer
        x = self.quasi(input)

        # Project identity if dimensions don't match
        if self.need_projection:
            identity = self.projection(identity)

        # Add the skip connection
        out = x + identity

        return out


if __name__ == '__main__':
    ...