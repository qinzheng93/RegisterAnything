import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from vision3d.layers.basic_layers import build_dropout_layer
from vision3d.loss.smooth_ce_loss import SmoothCrossEntropyLoss

from .functional import create_pat_conv1d_blocks, create_pat_conv2d_blocks, k_nearest_neighbors_graph

__all__ = [
    "AbsoluteRelativePositionEmbedding",
    "GroupShuffleAttention",
    "GumbelSubsetSampling",
    "AttentionSubsetSampling",
    "ElementwiseLoss",
]


class AbsoluteRelativePositionEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_dims: list[int],
        global_dims: list[int],
        num_neighbors: int,
        dilation: int = 1,
        ignore_nearest: bool = True,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.ignore_nearest = ignore_nearest
        self.dilation = dilation

        layers = create_pat_conv2d_blocks(input_dim, local_dims, groups=8)
        self.local_encoder = nn.Sequential(OrderedDict(*layers))

        input_dim = local_dims[-1]
        layers = create_pat_conv1d_blocks(input_dim, global_dims, groups=8)
        self.global_encoder = nn.Sequential(OrderedDict(*layers))

    def forward(self, points: Tensor) -> Tensor:
        neighbors = k_nearest_neighbors_graph(
            points,
            self.num_neighbors,
            dilation=self.dilation,
            training=self.training,
            ignore_nearest=self.ignore_nearest,
        )
        num_neighbors = neighbors.shape[3]
        points = points.unsqueeze(3).repeat(1, 1, 1, num_neighbors)
        points = torch.cat([points, neighbors - points], dim=1)
        points = self.local_encoder(points)
        points, _ = points.max(dim=3)
        points = self.global_encoder(points)
        return points


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float | None = None):
        super().__init__()
        self.dropout = build_dropout_layer(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        num_channels = q.shape[-1]
        attention = torch.matmul(q, k) / math.sqrt(num_channels)
        attention = torch.softmax(attention, dim=-2)
        attention = self.dropout(attention)
        v = torch.matmul(v, attention)
        return v


class GroupShuffleAttention(nn.Module):
    def __init__(self, feature_dim: int, groups: int, dropout: float | None = None):
        super().__init__()
        self.num_group = groups
        self.transform = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, groups=groups)
        self.attention = ScaledDotProductAttention(dropout)
        self.gn = nn.GroupNorm(groups, feature_dim)

    def forward(self, points: Tensor) -> Tensor:
        identity = points
        batch_size, num_channels, num_points = points.shape
        num_channels_per_group = num_channels // self.num_group
        points = self.transform(points)
        points = points.view(batch_size, self.num_group, num_channels_per_group, num_points)
        points = self.attention(points.transpose(2, 3), points, F.elu(points))
        points = points.transpose(1, 2).contiguous().view(batch_size, num_channels, num_points)
        points += identity
        points = self.gn(points)

        return points


class GumbelSubsetSampling(nn.Module):
    """
    Gumbel Subset Sampling proposed in `Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling`.

    Use soft sampling in training and hard sampling in testing.
    """

    def __init__(self, input_dim: int, num_samples: int, tau: float = 1.0):
        super().__init__()
        self.num_samples = num_samples
        self.tau = tau
        self.layer = nn.Conv1d(input_dim, num_samples, kernel_size=1)

    def forward(self, points: Tensor) -> Tensor:
        weight = self.layer(points)
        weight = F.gumbel_softmax(weight, tau=self.tau, hard=self.hard, dim=2).transpose(1, 2)
        points = torch.matmul(points, weight)
        return points

    @property
    def hard(self):
        return not self.training


class AttentionSubsetSampling(nn.Module):
    def __init__(self, input_dim: int, num_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.layer = nn.Conv1d(input_dim, num_samples, kernel_size=1)

    def forward(self, points: Tensor) -> Tensor:
        weight = self.layer(points)
        weight = torch.softmax(weight, dim=2).transpose(1, 2)
        points = torch.matmul(points, weight)
        return points


class ElementwiseLoss(nn.Module):
    def __init__(self, label_smoothing_eps: float | None = None):
        super().__init__()
        if label_smoothing_eps is None:
            self.cls_loss = nn.CrossEntropyLoss()
        else:
            self.cls_loss = SmoothCrossEntropyLoss(eps=label_smoothing_eps)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        num_points = inputs.shape[2]
        targets = targets.unsqueeze(1).repeat(1, num_points)
        cls_loss = self.cls_loss(inputs, targets)
        return cls_loss
