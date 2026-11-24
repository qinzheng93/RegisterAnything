from typing import Any, overload

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


@overload
def tensor_to_array(x: list) -> list: ...


@overload
def tensor_to_array(x: dict) -> dict: ...


@overload
def tensor_to_array(x: tuple) -> tuple: ...


@overload
def tensor_to_array(x: Tensor) -> ndarray | float | int: ...


def tensor_to_array(x: Any) -> Any:
    """Release all pytorch tensors to item or numpy arrays."""
    if isinstance(x, list):
        x = [tensor_to_array(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([tensor_to_array(item) for item in x])
    elif isinstance(x, dict):
        x = {key: tensor_to_array(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


@overload
def array_to_tensor(x: list) -> list: ...


@overload
def array_to_tensor(x: dict) -> dict: ...


@overload
def array_to_tensor(x: tuple) -> tuple: ...


@overload
def array_to_tensor(x: ndarray) -> Tensor: ...


def array_to_tensor(x: Any) -> Any:
    """Convert all numpy arrays to pytorch tensors."""
    if isinstance(x, list):
        x = [array_to_tensor(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([array_to_tensor(item) for item in x])
    elif isinstance(x, dict):
        x = {key: array_to_tensor(value) for key, value in x.items()}
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


@overload
def move_to_cpu(x: list) -> list: ...


@overload
def move_to_cpu(x: dict) -> dict: ...


@overload
def move_to_cpu(x: tuple) -> tuple: ...


@overload
def move_to_cpu(x: Tensor) -> Tensor: ...


def move_to_cpu(x: Any) -> Any:
    """Move all tensors to cpu."""
    if isinstance(x, list):
        x = [move_to_cpu(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([move_to_cpu(item) for item in x])
    elif isinstance(x, dict):
        x = {key: move_to_cpu(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = x.cpu()
    return x


@overload
def move_to_cuda(x: list) -> list: ...


@overload
def move_to_cuda(x: dict) -> dict: ...


@overload
def move_to_cuda(x: tuple) -> tuple: ...


@overload
def move_to_cuda(x: Tensor) -> Tensor: ...


def move_to_cuda(x: Any) -> Any:
    """Move all tensors to cuda."""
    if isinstance(x, list):
        x = [move_to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([move_to_cuda(item) for item in x])
    elif isinstance(x, dict):
        x = {key: move_to_cuda(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = x.cuda()
    return x
