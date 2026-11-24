import warnings

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from vision3d.layers.basic_layers import build_dropout_layer, build_norm_layer
from vision3d.utils.types import LayerCfg

from .common import AttentionOutput


class FlashAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
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

        if dropout is None:
            dropout = 0.0
        self.dropout = dropout

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        q_embeds: Tensor | None = None,
        k_embeds: Tensor | None = None,
        v_embeds: Tensor | None = None,
        k_masks: Tensor | None = None,
        qk_masks: Tensor | None = None,
    ) -> Tensor:
        """Multi-Head Attention forward propagation.

        Args:
            q_tokens (Tensor): query tokens (B, N, C)
            k_tokens (Tensor): key tokens (B, M, C)
            v_tokens (Tensor): value tokens (B, M, C)
            q_embeds (Tensor): query embeddings (B, N, C)
            k_embeds (Tensor): key embeddings (B, M, C)
            v_embeds (Tensor): value embeddings (B, M, C)
            qk_masks (Tensor): query-key bool masks. If False, ignored. (B, N, M)

        Returns:
            hidden_tokens (Tensor): output tokens (B, N, C)
        """
        # input check
        if self.q_embed_proj and q_embeds is None:
            warnings.warn("No 'q_embeds' but 'q_embed_proj' is set.")
        if self.k_embed_proj and k_embeds is None:
            warnings.warn("No 'k_embeds' but 'k_embed_proj' is set.")
        if self.v_embed_proj and v_embeds is None:
            warnings.warn("No 'v_embeds' but 'v_embed_proj' is set.")

        num_q_tokens = q_tokens.shape[1]

        # compute Q, K, V tokens
        q_tokens = self.q_token_layer(q_tokens)
        if q_embeds is not None:
            q_embeds = self.q_embed_layer(q_embeds)
            q_tokens = q_tokens + q_embeds  # type: ignore
        q_tokens = rearrange(q_tokens, "b n (h c) -> b h n c", h=self.num_heads)

        k_tokens = self.k_token_layer(k_tokens)
        if k_embeds is not None:
            k_embeds = self.k_embed_layer(k_embeds)
            k_tokens = k_tokens + k_embeds  # type: ignore
        k_tokens = rearrange(k_tokens, "b m (h c) -> b h m c", h=self.num_heads)

        v_tokens = self.v_token_layer(v_tokens)
        if v_embeds is not None:
            v_embeds = self.v_embed_layer(v_embeds)
            v_tokens = v_tokens + v_embeds  # type: ignore
        v_tokens = rearrange(v_tokens, "b m (h c) -> b h m c", h=self.num_heads)

        # reverse masks
        if k_masks is not None:
            k_masks = k_masks.unsqueeze(1).expand(-1, num_q_tokens, -1)
            qk_masks = torch.logical_and(qk_masks, k_masks) if qk_masks is not None else k_masks

        if qk_masks is not None:
            qk_masks = qk_masks.unsqueeze(1)

        # compute flash attention
        hidden_tokens = F.scaled_dot_product_attention(
            q_tokens, k_tokens, v_tokens, attn_mask=qk_masks, dropout_p=self.dropout
        )  # (b h n c)

        # reshape output
        hidden_tokens = rearrange(hidden_tokens, "b h n c -> b n (h c)")

        return hidden_tokens


class FlashAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        dropout: float | None = None,
        pre_norm: bool = False,
        norm_cfg: LayerCfg = "LayerNorm",
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = FlashAttention(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
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
        q_embeds: Tensor | None = None,
        k_embeds: Tensor | None = None,
        v_embeds: Tensor | None = None,
        k_masks: Tensor | None = None,
        qk_masks: Tensor | None = None,
    ) -> Tensor:
        hidden_tokens = q_tokens
        if self.pre_norm:
            hidden_tokens = self.norm(hidden_tokens)
        hidden_tokens = self.attention(
            hidden_tokens,
            k_tokens,
            v_tokens,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            k_masks=k_masks,
            qk_masks=qk_masks,
        )
        hidden_tokens = self.linear(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = hidden_tokens + q_tokens
        if not self.pre_norm:
            output_tokens = self.norm(output_tokens)
        return output_tokens


class FastTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        dropout: float | None = None,
        pre_norm: bool = False,
        norm_cfg: LayerCfg = "LayerNorm",
        act_cfg: LayerCfg = "ReLU",
    ):
        super().__init__()
        self.attention = FlashAttentionLayer(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
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
        q_embeds: Tensor | None = None,
        k_embeds: Tensor | None = None,
        v_embeds: Tensor | None = None,
        k_masks: Tensor | None = None,
        qk_masks: Tensor | None = None,
    ) -> Tensor:
        hidden_tokens = self.attention(
            q_tokens,
            k_tokens,
            v_tokens,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            k_masks=k_masks,
            qk_masks=qk_masks,
        )
        output_tokens = self.output(hidden_tokens)

        return output_tokens
