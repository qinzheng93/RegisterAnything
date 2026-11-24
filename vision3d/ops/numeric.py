import torch
from torch import Tensor


def safe_divide(a: Tensor | float, b: Tensor, eps: float = 1e-6) -> Tensor:
    b = torch.clamp(b, min=eps)
    return a / b


def safe_sqrt(a: Tensor, eps: float = 1e-6) -> Tensor:
    a = torch.clamp(a, min=eps)
    return torch.sqrt(a)
