import torch.nn as nn
from torch import Tensor

from vision3d.layers.basic_layers import build_act_layer, build_dropout_layer, build_norm_layer
from vision3d.utils.types import LayerCfg


class AttentionOutput(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float | None = None,
        pre_norm: bool = False,
        norm_cfg: LayerCfg = "LayerNorm",
        act_cfg: LayerCfg = "ReLU",
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(act_cfg)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = build_norm_layer(d_model, norm_cfg)

    def forward(self, input_tokens: Tensor) -> Tensor:
        hidden_tokens = input_tokens
        if self.pre_norm:
            hidden_tokens = self.norm(hidden_tokens)
        hidden_tokens = self.expand(hidden_tokens)
        hidden_tokens = self.activation(hidden_tokens)
        hidden_tokens = self.squeeze(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = input_tokens + hidden_tokens
        if not self.pre_norm:
            output_tokens = self.norm(output_tokens)
        return output_tokens
