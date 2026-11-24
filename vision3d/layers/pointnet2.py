import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops.group_gather import group_gather
from vision3d.utils.types import LayerCfg

from .conv_block import ConvBlock


class SAConv(nn.Module):
    """Set Abstraction Convolution in batch mode.

    Proposed in PointNet++.
    """

    def __init__(
        self, input_dim: int, output_dims: list[int], norm_cfg: LayerCfg = "BatchNorm", act_cfg: LayerCfg = "ReLU"
    ):
        super().__init__()
        layers = []
        for output_dim in output_dims:
            layers.append(
                ConvBlock(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=1,
                    conv_cfg="Conv2d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            input_dim = output_dim
        self.local_mlp = nn.Sequential(*layers)

    def forward(self, q_points: Tensor, s_points: Tensor, s_feats: Tensor, neighbor_indices: Tensor) -> Tensor:
        """Set Abstract forward.

        Args:
            q_points (Tensor): (B, 3, M)
            s_points (Tensor): (B, 3, N)
            s_feats (Tensor): (B, C_i, N)
            neighbor_indices (LongTensor): (B, N, K)

        Returns:
            q_feats (Tensor): (B, C_o, M)
        """
        neighbor_points = group_gather(s_points, neighbor_indices)  # (B, 3, N) -> (B, 3, M, K)
        neighbor_offsets = neighbor_points - q_points.unsqueeze(3)  # (B, 3, M, K)
        neighbor_feats = group_gather(s_feats, neighbor_indices)  # (B, C_i, N) -> (B, C_i, M, K)
        neighbor_feats = torch.cat([neighbor_feats, neighbor_offsets], dim=1)
        neighbor_feats = self.local_mlp(neighbor_feats)
        q_feats, _ = neighbor_feats.max(dim=3)
        return q_feats


class GSAConv(nn.Module):
    """Global Set Abstraction Convolution in batch mode.

    Proposed in PointNet++.
    """

    def __init__(
        self, input_dim: int, output_dims: list[int], norm_cfg: LayerCfg = "BatchNorm", act_cfg: LayerCfg = "ReLU"
    ):
        super().__init__()
        layers = []
        for output_dim in output_dims:
            layers.append(
                ConvBlock(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=1,
                    conv_cfg="Conv1d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            input_dim = output_dim
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, points: Tensor, feats: Tensor) -> Tensor:
        feats = torch.cat([feats, points], dim=1)
        feats = self.shared_mlp(feats)
        feats = feats.max(dim=2)[0]
        return feats
