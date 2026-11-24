from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers.transformer import FastTransformerLayer, TransformerLayer
from vision3d.utils.types import LayerCfg

from .geometric_structure_embedding import (
    GeometricStructureEmbedding,
    GeometricStructureEmbeddingV2,
    PairwiseDistanceEmbedding,
)

type AttentionType = Literal["geometric_self", "vanilla_self", "ordered_cross", "symmetric_cross", "global"]


def get_global_masks(
    src_tokens: Tensor, tgt_tokens: Tensor, src_masks: Tensor | None, tgt_masks: Tensor | None
) -> Tensor | None:
    # no masks
    if src_masks is None and tgt_masks is None:
        return None

    # compute all masks
    batch_size = src_tokens.shape[0]
    src_n_tokens = src_tokens.shape[1]
    tgt_n_tokens = tgt_tokens.shape[1]

    all_masks = torch.zeros(batch_size, src_n_tokens + tgt_n_tokens, dtype=torch.bool).cuda()
    if src_masks is not None:
        all_masks[:, :src_n_tokens] = src_masks
    if tgt_masks is not None:
        all_masks[:, src_n_tokens:] = tgt_masks

    return all_masks


class GeometricTransformer(nn.Module):
    """Geometric Transformer Module.

    Compute point features with attentions iteratively. In the current version, five types of attentions
    are supported:
    1. "geometric_self": geometric self-attention, which uses geometric structure embedding as the relative positional embedding.
    2. "vanilla_self": vanilla self-attention, which omits the relative positional embedding
    3. "ordered_cross": ordered cross-attention, which first uses the OLD source tokens as Q and the OLD target tokens
       as K/V to compute the new source tokens, then uses the OLD target tokens as Q and the NEW source tokens as K/V to
       compute the new target tokens.
    4. "symmetric_cross": symmetric cross-attention, which first uses the OLD source tokens as Q and the OLD target tokens
       as K/V to compute the new source tokens, then uses the OLD target tokens as Q and the OLD source tokens as K/V to
       compute the new target tokens.
    5. "global": global attention, which computes the self-attention on all tokens to generate the output tokens.

    We empirically note that, "ordered_cross" performs better than "symmetric_cross" and "global", and the latter two
    require warming up and smaller learning rate during training.

    There are five variants of Geometric Transformer:
    1. "GSEv1": The standard version, which uses embedding projections in all geometric self-attention.
    2. "GSEv2": The lite version, which moves all embedding projections in the attention module to the embedding computation,
       which reduces the computation overhead.
    3. "PDE": The version which uses only the pair-wise distance embedding.

    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        hidden_dim: hidden feature dimension
        num_heads: number of head in transformer
        blocks: type of attentions, available: ["geometric_self", "vanilla_self", "ordered_cross", "sym_cross", "global"]
        sigma_d: temperature of distance
        sigma_a: temperature of angles
        angle_k: number of nearest neighbors for angular embedding
        angle_reduction: reduction mode of angular embedding ["max", "mean"]
        dropout: dropout ratio in transformer
        activation_fn: activation function
        embedding: the geometric structure embedding ["GSEv1", "GSEv2", "PDE"]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        blocks: list[AttentionType],
        sigma_d: float,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction_a: Literal["max", "mean"] = "max",
        dropout: float | None = None,
        norm_cfg: LayerCfg = "LayerNorm",
        act_cfg: LayerCfg = "ReLU",
        embedding: Literal["GSEv1", "GSEv2", "PDE"] = "GSEv1",
    ):
        super().__init__()

        self.version = embedding

        match embedding:
            case "GSEv1":
                self.embedding = GeometricStructureEmbedding(
                    hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a
                )
            case "GSEv2":
                self.embedding = GeometricStructureEmbeddingV2(
                    hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a
                )
            case "PDE":
                self.embedding = PairwiseDistanceEmbedding(hidden_dim, sigma_d)
            case _:
                raise ValueError(f"Unknown embedding: {embedding}")

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.blocks = blocks
        layers = []
        for block in self.blocks:
            assert block in ["geometric_self", "vanilla_self", "ordered_cross", "symmetric_cross", "global"]
            if block == "geometric_self":
                qk_embed_proj = embedding in ["GSEv1", "PDE"]
                layers.append(
                    TransformerLayer(
                        hidden_dim,
                        num_heads,
                        qk_embed_proj=qk_embed_proj,
                        dropout=dropout,
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
        # compute embeddings
        src_embeddings = self.embedding(src_points)
        tgt_embeddings = self.embedding(tgt_points)

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
                    src_tokens, src_tokens, src_tokens, qk_embeds=src_embeddings, k_masks=src_masks
                )
                tgt_tokens = self.transformer[i](
                    tgt_tokens, tgt_tokens, tgt_tokens, qk_embeds=tgt_embeddings, k_masks=tgt_masks
                )
            elif block == "vanilla_self":
                # vanilla self-attention
                src_tokens = self.transformer[i](src_tokens, src_tokens, src_tokens, k_masks=src_masks)
                tgt_tokens = self.transformer[i](tgt_tokens, tgt_tokens, tgt_tokens, k_masks=tgt_masks)
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
