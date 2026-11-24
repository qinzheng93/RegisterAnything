import warnings

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from vision3d.layers.basic_layers import build_dropout_layer, build_norm_layer
from vision3d.utils.types import LayerCfg

from .common import AttentionOutput


class LocalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: float | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, f"'d_model={d_model}' is not divisible by 'num_heads={num_heads}'."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.q_token_layer = nn.Linear(d_model, d_model)
        self.k_token_layer = nn.Linear(d_model, d_model)
        self.v_token_layer = nn.Linear(d_model, d_model)

        self.q_embed_proj = q_embed_proj
        self.q_embed_layer = nn.Linear(d_model, d_model) if q_embed_proj else nn.Identity()

        self.k_embed_proj = k_embed_proj
        self.k_embed_layer = nn.Linear(d_model, d_model) if k_embed_proj else nn.Identity()

        self.v_embed_proj = v_embed_proj
        self.v_embed_layer = nn.Linear(d_model, d_model) if v_embed_proj else nn.Identity()

        self.qk_embed_proj = qk_embed_proj
        self.qk_embed_layer = nn.Linear(d_model, d_model) if qk_embed_proj else nn.Identity()

        self.qv_embed_proj = qv_embed_proj
        self.qv_embed_layer = nn.Linear(d_model, d_model) if qv_embed_proj else nn.Identity()

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        neighbor_indices: Tensor,
        q_embeds: Tensor | None = None,
        k_embeds: Tensor | None = None,
        v_embeds: Tensor | None = None,
        qk_embeds: Tensor | None = None,
        qv_embeds: Tensor | None = None,
        k_weights: Tensor | None = None,
        k_masks: Tensor | None = None,
        qk_weights: Tensor | None = None,
        qk_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Local Multi-Head Attention forward propagation.

        TODO: neighor masks support.

        Args:
            q_tokens (Tensor): query tokens (B, N, C)
            k_tokens (Tensor): key tokens (B, M, C)
            v_tokens (Tensor): value tokens (B, M, C)
            neighbor_indices (Tensor): neighbor indices (B, N, K)
            q_embeds (Tensor, optional): query embeddings (B, N, C)
            k_embeds (Tensor, optional): key embeddings (B, M, C)
            v_embeds (Tensor, optional): value embeddings (B, M, C)
            qk_embeds (Tensor, optional): query-key embeddings (B, N, K, C)
            qv_embeds (Tensor, optional): query-value embeddings (B, N, K, C)
            k_weights (Tensor, optional): key weights (B, M)
            k_masks (Tensor[bool], optional): key masks. If False, ignored. (B, M)
            qk_weights (Tensor, optional): query-key weights (B, N, K)
            qk_masks (Tensor[bool], optional): query-key masks. If False, ignored. (B, N, K)

        Returns:
            hidden_tokens (Tensor): output tokens (B, C, N)
            attention_scores (Tensor): attention scores (after dropout) (B, H, N, M)
        """
        # input check
        if self.q_embed_proj and q_embeds is None:
            warnings.warn("No 'q_embeds' but 'q_embed_proj' is set.")
        if self.k_embed_proj and k_embeds is None:
            warnings.warn("No 'k_embeds' but 'k_embed_proj' is set.")
        if self.v_embed_proj and v_embeds is None:
            warnings.warn("No 'v_embeds' but 'v_embed_proj' is set.")
        if self.qk_embed_proj and qk_embeds is None:
            warnings.warn("No 'qk_embeds' but 'qk_embed_proj' is set.")
        if self.qv_embed_proj and qv_embeds is None:
            warnings.warn("No 'qv_embeds' but 'qv_embed_proj' is set.")

        # expand neighbors
        num_q_tokens = neighbor_indices.shape[1]
        neighbor_token_indices = neighbor_indices.unsqueeze(3).expand(-1, -1, -1, self.d_model)  # (B, N, K, C)

        # compute query tokens
        q_tokens = self.q_token_layer(q_tokens)  # (B, N, C)
        if q_embeds is not None:
            q_embeds = self.q_embed_layer(q_embeds)
            q_tokens = q_tokens + q_embeds  # type: ignore
        q_tokens = rearrange(q_tokens, "b n (h c) -> b n h c", h=self.num_heads)

        # compute key tokens
        k_tokens = self.k_token_layer(k_tokens)  # (B, M, C)
        if k_embeds is not None:
            k_embeds = self.k_embed_layer(k_embeds)
            k_tokens = k_tokens + k_embeds  # type: ignore
        k_tokens = k_tokens.unsqueeze(1).expand(-1, num_q_tokens, -1, -1)  # (B, N, M, C)
        k_tokens = torch.gather(k_tokens, dim=2, index=neighbor_token_indices)  # (B, N, K, C)
        k_tokens = rearrange(k_tokens, "b n k (h c) -> b n h k c", h=self.num_heads)

        if qk_embeds is not None:
            qk_embeds = self.qk_embed_layer(qk_embeds)  # (B, N, K, C)
            qk_embeds = rearrange(qk_embeds, "b n k (h c) -> b n h k c", h=self.num_heads)
            k_tokens = k_tokens + qk_embeds  # type: ignore

        # compute value tokens
        v_tokens = self.v_token_layer(v_tokens)  # (B, M, C)
        if v_embeds is not None:
            v_embeds = self.v_embed_layer(v_embeds)
            v_tokens = v_tokens + v_embeds  # type: ignore
        v_tokens = v_tokens.unsqueeze(1).expand(-1, num_q_tokens, -1, -1)  # (B, N, M, C)
        v_tokens = torch.gather(v_tokens, dim=2, index=neighbor_token_indices)  # (B, N, K, C)
        v_tokens = rearrange(v_tokens, "b n k (h c) -> b n h k c", h=self.num_heads)

        if qv_embeds is not None:
            qv_embeds = self.qv_embed_layer(qv_embeds)
            qv_embeds = rearrange(qv_embeds, "b n k (h c) -> b n h k c", h=self.num_heads)
            v_tokens = v_tokens + qv_embeds  # type: ignore

        # merge k_weights and qk_weights
        if k_weights is not None:
            k_weights = k_weights.unsqueeze(1).expand(-1, num_q_tokens, -1)  # (B, N, M)
            k_weights = torch.gather(k_weights, dim=2, index=neighbor_indices)  # (B, N, K)
            qk_weights = qk_weights * k_weights if qk_weights is not None else k_weights

        # merge k_masks and qk_masks
        if k_masks is not None:
            k_masks = k_masks.unsqueeze(1).expand(-1, num_q_tokens, -1)  # (B, N, M)
            k_masks = torch.gather(k_masks, dim=2, index=neighbor_indices)  # (B, N, K)
            qk_masks = torch.logical_and(qk_masks, k_masks) if qk_masks is not None else k_masks  # (B, N, K)

        # TODO: use SDPA when there are no qk_weights

        # compute attention scores
        attention_scores = torch.einsum("bnhc,bnhkc->bnhk", q_tokens, k_tokens)
        attention_scores = attention_scores / self.d_model_per_head**0.5

        # handle weights
        if qk_weights is not None:
            attention_scores = attention_scores * qk_weights.unsqueeze(2)

        # handle masks
        if qk_masks is not None:
            qk_masks = qk_masks.logical_not()
            attention_scores = attention_scores.masked_fill(qk_masks.unsqueeze(2), float("-inf"))

        # compute attention probabilities
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        # compute output tokens
        hidden_tokens = torch.einsum("bnhk,bnhkc->bnhc", attention_scores, v_tokens)
        hidden_tokens = rearrange(hidden_tokens, "b n h c -> b n (h c)")

        return hidden_tokens, attention_scores


class LocalAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: float | None = None,
        pre_norm: bool = False,
        norm_cfg: LayerCfg = "LayerNorm",
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = LocalMultiHeadAttention(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
            qk_embed_proj=qk_embed_proj,
            qv_embed_proj=qv_embed_proj,
            dropout=dropout,
        )
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = build_norm_layer(d_model, norm_cfg)

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        neighbor_indices: Tensor,
        q_embeds: Tensor | None = None,
        k_embeds: Tensor | None = None,
        v_embeds: Tensor | None = None,
        qk_embeds: Tensor | None = None,
        qv_embeds: Tensor | None = None,
        k_weights: Tensor | None = None,
        k_masks: Tensor | None = None,
        qk_weights: Tensor | None = None,
        qk_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden_tokens = q_tokens
        if self.pre_norm:
            hidden_tokens = self.norm(hidden_tokens)
        hidden_tokens, attention_scores = self.attention(
            hidden_tokens,
            k_tokens,
            v_tokens,
            neighbor_indices,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            qk_embeds=qk_embeds,
            qv_embeds=qv_embeds,
            k_weights=k_weights,
            k_masks=k_masks,
            qk_weights=qk_weights,
            qk_masks=qk_masks,
        )
        hidden_tokens = self.linear(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = hidden_tokens + q_tokens
        if not self.pre_norm:
            output_tokens = self.norm(output_tokens)
        return output_tokens, attention_scores


class LocalTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: float | None = None,
        pre_norm: bool = False,
        norm_cfg: LayerCfg = "LayerNorm",
        act_cfg: LayerCfg = "ReLU",
    ):
        super().__init__()
        self.attention = LocalAttentionLayer(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
            qk_embed_proj=qk_embed_proj,
            qv_embed_proj=qv_embed_proj,
            dropout=dropout,
            pre_norm=pre_norm,
            norm_cfg=norm_cfg,
        )
        self.output = AttentionOutput(
            d_model,
            dropout=dropout,
            pre_norm=pre_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        neighbor_indices: Tensor,
        q_embeds: Tensor | None = None,
        k_embeds: Tensor | None = None,
        v_embeds: Tensor | None = None,
        qk_embeds: Tensor | None = None,
        qv_embeds: Tensor | None = None,
        k_weights: Tensor | None = None,
        k_masks: Tensor | None = None,
        qk_weights: Tensor | None = None,
        qk_masks: Tensor | None = None,
        return_attention_score: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        hidden_tokens, attention_scores = self.attention(
            q_tokens,
            k_tokens,
            v_tokens,
            neighbor_indices,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            qk_embeds=qk_embeds,
            qv_embeds=qv_embeds,
            k_weights=k_weights,
            k_masks=k_masks,
            qk_weights=qk_weights,
            qk_masks=qk_masks,
        )
        output_tokens = self.output(hidden_tokens)

        if return_attention_score:
            return output_tokens, attention_scores
        return output_tokens
