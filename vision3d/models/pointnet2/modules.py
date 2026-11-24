import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers.pointnet2 import GSAConv, SAConv
from vision3d.ops.ball_query import ball_query
from vision3d.ops.furthest_point_sample import furthest_point_sample
from vision3d.utils.types import LayerCfg


class SetAbstractionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        num_centroids: int,
        num_samples: int,
        radius: float,
        norm_cfg: LayerCfg = "BatchNorm",
        act_cfg: LayerCfg = "ReLU",
    ):
        super().__init__()
        self.num_centroids = num_centroids
        self.num_samples = num_samples
        self.radius = radius
        self.set_abstract = SAConv(input_dim, output_dims, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, s_points: Tensor, s_feats: Tensor) -> tuple[Tensor, Tensor]:
        """Set Abstraction Module in PointNet++ (batch mode).

        Args:
            s_points (Tensor): The input points in shape of (B, 3, N).
            s_feats (Tensor): The features of the point in shape of (B, C_in, N).

        Returns:
            q_points (Tensor): The sampled point in shape of (B, 3, M).
            q_feats (Tensor): The features of the sampled points in shape of (B, C_out, M)
        """
        _, q_points = furthest_point_sample(s_points, self.num_centroids, transposed=True)
        neighbor_indices = ball_query(q_points, s_points, self.num_samples, self.radius, transposed=True)
        q_feats = self.set_abstract(q_points, s_points, s_feats, neighbor_indices)
        return q_points, q_feats


class MultiScaleSetAbstractionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_centroids: int,
        ssg_cfgs: list[dict],
        norm_cfg: LayerCfg = "BatchNorm",
        act_cfg: LayerCfg = "ReLU",
    ):
        super().__init__()

        self.ssg_cfgs = ssg_cfgs
        self.num_centroids = num_centroids

        layers = []
        for ssg_cfg in ssg_cfgs:
            layers.append(SAConv(input_dim, ssg_cfg["output_dims"], norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.ssg_layers = nn.ModuleList(layers)

    def forward(self, s_points: Tensor, s_feats: Tensor) -> tuple[Tensor, Tensor]:
        _, q_points = furthest_point_sample(s_points, self.num_centroids, transposed=True)
        feats_list = []
        for i, ssg_cfg in enumerate(self.ssg_cfgs):
            neighbor_indices = ball_query(
                q_points, s_points, ssg_cfg["num_samples"], ssg_cfg["radius"], transposed=True
            )
            ssg_feats = self.ssg_layers[i](q_points, s_points, s_feats, neighbor_indices)
            feats_list.append(ssg_feats)
        q_feats = torch.cat(feats_list, dim=1)
        return q_points, q_feats


class GlobalAbstractionModule(nn.Module):
    def __init__(self, input_dim: int, output_dims: list[int]):
        super().__init__()
        self.global_abstract = GSAConv(input_dim, output_dims)

    def forward(self, points: Tensor, feats: Tensor) -> Tensor:
        feats = self.global_abstract(points, feats).unsqueeze(2)
        return feats
