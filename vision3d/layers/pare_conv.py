from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .vector_neurons import VNLeakyReLU, VNLinear, VNLinearBlock


class ScoreNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        self.vn_proj = VNLinearBlock(input_dim, output_dim * 2)

        layers = []
        last_dim = output_dim * 2
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Conv1d(last_dim, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.InstanceNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Conv1d(last_dim, output_dim, kernel_size=1))
        self.score_proj = nn.Sequential(*layers)

    def forward(self, feats: Tensor) -> Tensor:
        """Forward pass of the CorrelationNet.

        Args:
            feats (Tensor): Input features with shape (N, C, 3, K).

        Returns:
            Tensor: Output features with shape (N, C, K).
        """
        feats = self.vn_proj(feats)  # (N, C', 3, K)
        feats = torch.linalg.norm(feats, dim=2)  # (N, C', K)
        scores = self.score_proj(feats)  # (N, D, K)
        scores = torch.softmax(scores / self.temperature, dim=1)
        return scores


class PAREConvStemBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, share_nonlinearity: bool = False):
        super().__init__()

        assert input_dim == 3, "The stem block only supports 3-d input."

        self.kernel_size = kernel_size
        self.score_net = ScoreNet(input_dim, kernel_size, [kernel_size])

        weight_bank = torch.zeros(size=(self.kernel_size, input_dim, output_dim // 2))  # (W, C, D/2)
        nn.init.kaiming_normal_(weight_bank)
        self.weight_bank = nn.Parameter(weight_bank)

        self.vn_act = VNLeakyReLU(output_dim // 2, share_nonlinearity=share_nonlinearity)
        self.vn_unary = VNLinearBlock(output_dim // 2, output_dim)

    def forward(self, q_points: Tensor, s_points: Tensor, neighbor_indices: Tensor) -> Tensor:
        """Forward pass of the PareConvBlock.

        Args:
            q_points (Tensor): Query points with shape (N, 3).
            s_points (Tensor): Support points with shape (M, 3).
            neighbor_indices (Tensor): Neighbor indices with shape (N, K).

        Returns:
            Tensor: Output features with shape (N, D, 3).
        """
        max_neighbors = neighbor_indices.shape[1]
        masks = torch.lt(neighbor_indices, s_points.shape[0]).float()  # (N, K)
        num_neighbors = torch.clamp(masks.sum(1), min=1.0)  # (N,)

        # compute local features
        padded_s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :])], 0)  # (N, 3) -> (N+1, 3)
        neighbor_points = padded_s_points[neighbor_indices]  # (N, K, 3)
        offsets = neighbor_points - q_points.unsqueeze(1)  # (N, K, 3)
        centroids = offsets.sum(dim=1, keepdim=True) / num_neighbors.view(-1, 1, 1)  # (N, 1, 3)
        centroids = centroids.repeat(1, max_neighbors, 1)  # (N, K, 3)
        cross_prods = torch.linalg.cross(offsets, centroids, dim=-1)  # (N, K, 3)
        local_geometries = torch.stack([offsets, centroids, cross_prods], dim=1)  # (N, C0, K, 3)
        local_geometries = local_geometries.permute(0, 1, 3, 2)  # (N, C0, 3, K)

        # compute convolutional kernel scores
        scores = self.score_net(local_geometries)  # (N, W, K)
        scores = scores * masks.unsqueeze(1)  # (N, W, K)

        # compute convolution
        weighted_feats = torch.einsum("nclk,nwk->nwcl", local_geometries, scores)  # (N, W, C0, 3)
        q_feats = torch.einsum("nwcl,wcd->ndl", weighted_feats, self.weight_bank)  # (N, D/2, 3)
        q_feats = q_feats / num_neighbors.view(-1, 1, 1)  # (N, D/2, 3)

        # compute convolution
        # local_feats = torch.einsum("nclk,wcd->nwdlk", local_geometries, self.weight_bank)  # (N, W, D/2, 3, K)
        # local_feats = torch.einsum("nwdlk,nwk->ndlk", local_feats, scores)  # (N, D/2, 3, K)

        # normalize and mean
        # local_feats = F.normalize(local_feats, dim=2, p=2)  # (N, D/2, 3, K)
        # q_feats = local_feats.sum(dim=-1) / num_val_neighbors.view(-1, 1, 1)  # (N, D/2, 3)

        # last vn layers
        q_feats = self.vn_act(q_feats)  # (N, D/2, 3)
        q_feats = self.vn_unary(q_feats)  # (N, D, 3)

        return q_feats


class PAREConvResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        shared_nonlinearity: bool = False,
        mode: Literal["edge", "node"] = "edge",
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.mode = mode

        self.score_net = ScoreNet(3, kernel_size, [kernel_size])

        w_input_dim = input_dim if mode == "node" else 2 * input_dim + 1
        weight_bank = torch.zeros(size=(self.kernel_size, w_input_dim, output_dim // 2))  # (W, C, D/2)
        nn.init.kaiming_normal_(weight_bank)
        self.weight_bank = nn.Parameter(weight_bank)

        if input_dim != output_dim:
            self.shortcut = VNLinear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

        self.vn_act = VNLeakyReLU(output_dim // 2, share_nonlinearity=shared_nonlinearity)
        self.vn_unary = VNLinearBlock(output_dim // 2, output_dim)

    def forward(self, q_points: Tensor, s_points: Tensor, s_feats: Tensor, neighbor_indices: Tensor) -> Tensor:
        """Forward pass of the PareConvBlock.

        Args:
            q_points (Tensor): Query points with shape (N, 3).
            s_points (Tensor): Support points with shape (M, 3).
            s_feats (Tensor): Support features with shape (M, D, 3).
            neighbor_indices (Tensor): Neighbor indices with shape (N, K).

        Returns:
            Tensor: Output features with shape (N, D, 3).
        """
        max_neighbors = neighbor_indices.shape[1]
        masks = torch.lt(neighbor_indices, s_points.shape[0]).float()  # (N, K)
        num_neighbors = torch.clamp(masks.sum(1), min=1.0)  # (N,)

        # compute local features
        padded_s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :])], 0)  # (M, 3) -> (M+1, 3)
        neighbor_points = padded_s_points[neighbor_indices]  # (N, K, 3)
        offsets = neighbor_points - q_points.unsqueeze(1)  # (N, K, 3)
        centroids = offsets.sum(dim=1, keepdim=True) / num_neighbors.view(-1, 1, 1)  # (N, 1, 3)
        centroids = centroids.repeat(1, max_neighbors, 1)  # (N, K, 3)
        cross_prods = torch.linalg.cross(offsets, centroids, dim=-1)  # (N, K, 3)
        local_geometries = torch.stack([offsets, centroids, cross_prods], dim=1)  # (N, C0, K, 3)
        local_geometries = local_geometries.permute(0, 1, 3, 2)  # (N, C0, 3, K)

        # compute convolutional kernel scores
        scores = self.score_net(local_geometries)  # (N, W, K)
        scores = scores * masks.unsqueeze(1)  # (N, W, K)

        # construct local features
        padded_s_feats = torch.cat([s_feats, torch.zeros_like(s_feats[:1, :])], 0)  # (M, 3) -> (M+1, 3)
        local_feats = padded_s_feats[neighbor_indices]  # (N, K, C, 3)
        local_feats = local_feats.permute(0, 2, 3, 1)  # (N, C, 3, K)
        nearest_feats = local_feats[..., 0]

        # compute shortcut
        shortcut = self.shortcut(nearest_feats)  # (N, D, 3)

        # construct edge features
        if self.mode == "edge":
            offsets = offsets.unsqueeze(1).permute(0, 1, 3, 2)  # (N, 1, 3, K)
            offset_feats = local_feats - nearest_feats.unsqueeze(-1)  # (N, C, 3, K)
            local_feats = torch.cat([local_feats, offset_feats, offsets], dim=1)  # (N, 2*C+1, 3, K)

        # compute convolution
        weighted_feats = torch.einsum("nclk,nwk->nwcl", local_feats, scores)  # (N, W, C0, 3)
        residual = torch.einsum("nwcl,wcd->ndl", weighted_feats, self.weight_bank)  # (N, D/2, 3)
        residual = residual / num_neighbors.view(-1, 1, 1)  # (N, D/2, 3)

        # compute convolutional weights
        # local_feats = torch.einsum("nclk,wcd->nwdlk", local_feats, self.weight_bank)  # (N, W, D/2, 3, K)
        # local_feats = torch.einsum("nwdlk,nwk->ndlk", local_feats, scores)  # (N, D/2, 3, K)

        # normalize and mean
        # local_feats = F.normalize(local_feats, dim=2, p=2)  # (N, D/2, 3, K)
        # residual = local_feats.sum(dim=-1) / num_neighbors.view(-1, 1, 1)  # (N, D/2, 3)

        # last vn layers
        residual = self.vn_act(residual)  # (N, D/2, 3)
        residual = self.vn_unary(residual)  # (N, D, 3)

        # add shortcut
        q_feats = residual + shortcut

        return q_feats
