from collections import OrderedDict

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from vision3d.ops.group_gather import group_gather
from vision3d.ops.knn import knn


def create_pat_conv1d_blocks(
    input_dim: int, output_dims: list[int], groups: int, dropout: float | None = None
) -> list[Module]:
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ("conv", nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)),
            ("gn", nn.GroupNorm(groups, output_dim)),
            ("elu", nn.ELU(inplace=True)),
        ]
        if dropout is not None:
            block.append(("dp", nn.Dropout(dropout)))
        layers.append(("conv{}".format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def create_pat_conv2d_blocks(
    input_dim: int, output_dims: list[int], groups: int, dropout: float | None = None
) -> list[Module]:
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ("conv", nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)),
            ("gn", nn.GroupNorm(groups, output_dim)),
            ("elu", nn.ELU(inplace=True)),
        ]
        if dropout is not None:
            block.append(("dp", nn.Dropout(dropout)))
        layers.append(("conv{}".format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def create_pat_linear_blocks(
    input_dim: int, output_dims: list[int], groups: int, dropout: float | None = None
) -> list[Module]:
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ("fc", nn.Linear(input_dim, output_dim, bias=False)),
            ("gn", nn.GroupNorm(groups, output_dim)),
            ("elu", nn.ELU(inplace=True)),
        ]
        if dropout is not None:
            block.append(("dp", nn.Dropout(dropout)))
        layers.append(("fc{}".format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def k_nearest_neighbors_graph(
    points: Tensor, num_neighbors: int, dilation: int = 1, training: bool = True, ignore_nearest: bool = True
) -> Tensor:
    num_neighbors_dilated = num_neighbors * dilation + int(ignore_nearest)
    indices = knn(points, points, num_neighbors_dilated, transposed=True)
    start_index = 1 if ignore_nearest else 0
    if training:
        indices = indices[:, :, start_index::dilation].contiguous()
    else:
        indices = indices[:, :, start_index:].contiguous()
    points = group_gather(points, indices)
    return points
