import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()

        self.d_model = d_model

        self.eps = float(eps)

        self.device = device
        self.dtype = dtype
        self.device = device
        self.weight = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps).sqrt()
        return result.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to(self.weight.device)
        return x / self.rms(x) * self.weight


if __name__ == "__main__":
    pass
