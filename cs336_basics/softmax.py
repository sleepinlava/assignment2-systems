import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_tensor = x.max(dim=dim, keepdim=True).values
    up = torch.exp(x - max_tensor)
    down = torch.sum(torch.exp(x - max_tensor), dim=dim, keepdim=True)
    return up / down


if __name__ == "__main__":
    pass
