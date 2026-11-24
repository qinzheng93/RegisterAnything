from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers.transformer import FastTransformerLayer, LocalTransformerLayer
from vision3d.ops.knn import knn
from vision3d.utils.types import LayerCfg

from .geometric_transformer import AttentionType, get_global_masks
from .local_geometric_structure_embedding import (
    LocalGeometricStructureEmbedding,
    LocalGeometricStructureEmbeddingV2,
    LocalPairwiseDistanceEmbedding,
)


class LocalGeometricTransformer(nn.Module):
    """Local Geometric Transformer.

    The local version of Geometric Transformer, which only aggregates the tokens from neighborhood for each point in the
    self-attention. This slightly reduces the accuracy, but significantly improves the efficiency and memory usage.

    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        hidden_dim: hidden feature dimension
        num_heads: number of head in transformer
        k: number of neighbors in self-attention
        blocks: type of attentions, available: ["geometric_self", "vanilla_self", "ordered_cross", "sym_cross", "global"]
        sigma_d: temperature of distance
        sigma_a: temperature of angles
        angle_k: number of nearest neighbors for angular embedding
        angle_reduction: reduction mode of angular embedding ["max", "mean"]
        dropout: dropout ratio in transformer
        activation_fn: activation function
        version: the version of the geometric structure embedding ["v1", "v2"]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_neighbors: int,
        blocks: list[AttentionType],
        sigma_d: float,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction_a: Literal["max", "mean"] = "max",
        dropout: float | None = None,
        pre_norm: bool = False,
        norm_cfg: LayerCfg = "LayerNorm",
        act_cfg: LayerCfg = "ReLU",
        embedding: Literal["GSEv1", "GSEv2", "PDE"] = "GSEv1",
    ):
        super().__init__()

        self.num_neighbors = num_neighbors
        self.version = embedding

        match embedding:
            case "GSEv1":
                self.embedding = LocalGeometricStructureEmbedding(
                    hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a
                )
            case "GSEv2":
                self.embedding = LocalGeometricStructureEmbeddingV2(
                    hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a
                )
            case "PDE":
                self.embedding = LocalPairwiseDistanceEmbedding(hidden_dim, sigma_d)
            case _:
                raise ValueError(f"Unknown embedding: {embedding}")

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.blocks = blocks
        layers = []
        for block in self.blocks:
            assert block in ["geometric_self", "vanilla_self", "ordered_cross", "symmetric_cross", "global"]
            if block == "geometric_self" or block == "vanilla_self":
                qk_embed_proj = block == "geometric_self" and embedding in ["GSEv1", "PDE"]
                layers.append(
                    LocalTransformerLayer(
                        hidden_dim,
                        num_heads,
                        qk_embed_proj=qk_embed_proj,
                        dropout=dropout,
                        pre_norm=pre_norm,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
            else:
                layers.append(
                    FastTransformerLayer(
                        hidden_dim,
                        num_heads,
                        dropout=dropout,
                        pre_norm=pre_norm,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
        self.transformer = nn.ModuleList(layers)

        self.has_global_attention = "global" in self.blocks

    def forward(
        self,
        src_points: Tensor,
        tgt_points: Tensor,
        src_feats: Tensor,
        tgt_feats: Tensor,
        src_masks: Tensor | None = None,
        tgt_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Geometric Transformer

        Args:
            src_points (Tensor): (B, N, 3)
            tgt_points (Tensor): (B, M, 3)
            src_feats (Tensor): (B, N, C)
            tgt_feats (Tensor): (B, M, C)
            src_masks (Tensor[bool], optional): (B, N)
            tgt_masks (Tensor[bool], optional): (B, M)

        Returns:
            src_feats: torch.Tensor (B, N, C)
            tgt_feats: torch.Tensor (B, M, C)
        """
        src_num_points = src_points.shape[1]
        tgt_num_points = tgt_points.shape[1]

        # compute neighbors
        k = min(self.num_neighbors, min(src_num_points, tgt_num_points))
        src_neighbor_indices = knn(src_points, src_points, k)  # (B, N, k)
        tgt_neighbor_indices = knn(tgt_points, tgt_points, k)  # (B, M, k)

        # compute neighbor masks
        src_neighbor_masks = None
        if src_masks is not None:
            # For the real points, the real neighbors are True
            exp_src_rmasks = src_masks.unsqueeze(1).expand(-1, src_num_points, -1)  # (B, N, N)
            exp_src_rmasks = torch.logical_and(exp_src_rmasks, exp_src_rmasks.transpose(1, 2))
            src_neighbor_rmasks = torch.gather(exp_src_rmasks, dim=2, index=src_neighbor_indices)  # (B, N, k)
            # For the padded (virtual) points, the padded (virtual) neighbors are True
            src_vmasks = src_masks.logical_not()
            exp_src_vmasks = src_vmasks.unsqueeze(1).expand(-1, src_num_points, -1)  # (B, N, N)
            exp_src_vmasks = torch.logical_and(exp_src_vmasks, exp_src_vmasks.transpose(1, 2))
            src_neighbor_vmasks = torch.gather(exp_src_vmasks, dim=2, index=src_neighbor_indices)  # (B, N, k)
            src_neighbor_masks = torch.logical_or(src_neighbor_rmasks, src_neighbor_vmasks)  # (B, N, k)

        tgt_neighbor_masks = None
        if tgt_masks is not None:
            # For the real points, the real neighbors are True
            exp_tgt_rmasks = tgt_masks.unsqueeze(1).expand(-1, tgt_num_points, -1)  # (B, N, N)
            exp_tgt_rmasks = torch.logical_and(exp_tgt_rmasks, exp_tgt_rmasks.transpose(1, 2))
            tgt_neighbor_rmasks = torch.gather(exp_tgt_rmasks, dim=2, index=tgt_neighbor_indices)  # (B, N, k)
            # For the padded (virtual) points, the padded (virtual) neighbors are True
            tgt_vmasks = tgt_masks.logical_not()
            exp_tgt_vmasks = tgt_vmasks.unsqueeze(1).expand(-1, tgt_num_points, -1)  # (B, N, N)
            exp_tgt_vmasks = torch.logical_and(exp_tgt_vmasks, exp_tgt_vmasks.transpose(1, 2))
            tgt_neighbor_vmasks = torch.gather(exp_tgt_vmasks, dim=2, index=tgt_neighbor_indices)  # (B, N, k)
            tgt_neighbor_masks = torch.logical_or(tgt_neighbor_rmasks, tgt_neighbor_vmasks)  # (B, N, k)

        # compute embeddings
        src_embeddings = self.embedding(src_points, src_neighbor_indices)
        tgt_embeddings = self.embedding(tgt_points, tgt_neighbor_indices)

        # compute in_proj
        src_tokens = self.in_proj(src_feats)
        tgt_tokens = self.in_proj(tgt_feats)

        # compute all_masks for global attention
        all_masks = None
        if self.has_global_attention:
            all_masks = get_global_masks(src_tokens, tgt_tokens, src_masks=src_masks, tgt_masks=tgt_masks)

        # attention blocks
        for i, block in enumerate(self.blocks):
            if block == "geometric_self":
                # geometric self-attention
                src_tokens = self.transformer[i](
                    src_tokens,
                    src_tokens,
                    src_tokens,
                    src_neighbor_indices,
                    qk_embeds=src_embeddings,
                    qk_masks=src_neighbor_masks,
                )
                tgt_tokens = self.transformer[i](
                    tgt_tokens,
                    tgt_tokens,
                    tgt_tokens,
                    tgt_neighbor_indices,
                    qk_embeds=tgt_embeddings,
                    qk_masks=tgt_neighbor_masks,
                )
            elif block == "vanilla_self":
                # vanilla self-attention
                src_tokens = self.transformer[i](
                    src_tokens, src_tokens, src_tokens, src_neighbor_indices, k_masks=src_masks
                )
                tgt_tokens = self.transformer[i](
                    tgt_tokens, tgt_tokens, tgt_tokens, tgt_neighbor_indices, k_masks=tgt_masks
                )
            elif block == "ordered_cross":
                # ordered cross-attention
                src_tokens = self.transformer[i](src_tokens, tgt_tokens, tgt_tokens, k_masks=tgt_masks)
                tgt_tokens = self.transformer[i](tgt_tokens, src_tokens, src_tokens, k_masks=src_masks)
            elif block == "symmetric_cross":
                # symmetric cross-attention
                new_src_tokens = self.transformer[i](src_tokens, tgt_tokens, tgt_tokens, k_masks=tgt_masks)
                new_tgt_tokens = self.transformer[i](tgt_tokens, src_tokens, src_tokens, k_masks=src_masks)
                src_tokens = new_src_tokens
                tgt_tokens = new_tgt_tokens
            else:
                # global-attention
                src_n_tokens = src_tokens.shape[1]
                all_tokens = torch.cat([src_tokens, tgt_tokens], dim=1)  # (B, N+M, C)
                all_tokens = self.transformer[i](all_tokens, all_tokens, all_tokens, k_masks=all_masks)
                src_tokens = all_tokens[:, :src_n_tokens]
                tgt_tokens = all_tokens[:, src_n_tokens:]

        # compute out_proj
        src_feats = self.out_proj(src_tokens)
        tgt_feats = self.out_proj(tgt_tokens)

        return src_feats, tgt_feats
