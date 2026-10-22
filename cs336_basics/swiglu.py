import torch
from torch import nn, sigmoid

from cs336_basics import linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """x/(1 + exp(x))"""
    return x * sigmoid(x)


class positionwise_feedward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.weight_1 = linear.Linear(d_model, d_ff, device, dtype)
        self.weight_2 = linear.Linear(d_ff, d_model, device, dtype)
        self.weight_3 = linear.Linear(d_model, d_ff, device, dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """x/(1 + exp(x))"""
        return x * sigmoid(x)

    def forward(self, x: torch.Tensor):
        # swiglu = new FFN
        # here we combine the gate-control with activate function
        return self.weight_2.forward(self.silu(self.weight_1.forward(x)) * self.weight_3(x))


if __name__ == "__main__":
    pass
