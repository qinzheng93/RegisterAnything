import torch.nn as nn
from torch.nn import Module

from vision3d.utils.types import LayerCfg

from .depthwise_conv import DepthwiseConv1d, DepthwiseConv2d
from .norm import BatchNormPackMode, GroupNormPackMode, InstanceNormPackMode
from .separable_conv import SeparableConv1d, SeparableConv2d

NORM_LAYERS = {
    "None": nn.Identity,
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
    "RMSNorm": nn.RMSNorm,
}


NORM_LAYERS_PACK_MODE = {
    "None": nn.Identity,
    "BatchNorm": BatchNormPackMode,
    "InstanceNorm": InstanceNormPackMode,
    "GroupNorm": GroupNormPackMode,
    "LayerNorm": nn.LayerNorm,
}


ACT_LAYERS = {
    "None": nn.Identity,
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "Hardsigmoid": nn.Hardsigmoid,
    "Hardtanh": nn.Hardtanh,
    "Hardswish": nn.Hardswish,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "PReLU": nn.PReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "SiLU": nn.SiLU,
    "Mish": nn.Mish,
    "Softplus": nn.Softplus,
    "Softshrink": nn.Softshrink,
    "Tanh": nn.Tanh,
    "Tanhshrink": nn.Tanhshrink,
    "Threshold": nn.Threshold,
    "Identity": nn.Identity,
}


CONV_LAYERS = {
    "Linear": nn.Linear,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "DepthwiseConv1d": DepthwiseConv1d,
    "DepthwiseConv2d": DepthwiseConv2d,
    "SeparableConv1d": SeparableConv1d,
    "SeparableConv2d": SeparableConv2d,
}


def parse_cfg(cfg: LayerCfg) -> tuple[str, dict]:
    if isinstance(cfg, str):
        return cfg, {}
    assert isinstance(cfg, dict), "Illegal cfg type: {}.".format(type(cfg))
    layer = cfg["type"]
    kwargs = {key: value for key, value in cfg.items() if key != "type"}
    return layer, kwargs


def build_dropout_layer(p: float | None, **kwargs) -> Module:
    """Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    return nn.Dropout(p=p, **kwargs)


def _find_optimal_num_groups(num_channels: int) -> int:
    """Find the optimal number of groups for GroupNorm."""
    # strategy: at most 32 groups, at least 8 channels per group
    num_groups = 32
    while num_groups > 1:
        if num_channels % num_groups == 0:
            num_channels_per_group = num_channels // num_groups
            if num_channels_per_group >= 8:
                break
        num_groups = num_groups // 2
    assert num_groups != 1, (
        f"Cannot find 'num_groups' in GroupNorm with 'num_channels={num_channels}' automatically. "
        "Please manually specify 'num_groups'."
    )
    return num_groups


def _configure_norm_args(layer: str, kwargs: dict, normalized_shape: int) -> dict:
    """Configure norm args."""
    if layer == "GroupNorm":
        kwargs["num_channels"] = normalized_shape
        if "num_groups" not in kwargs:
            kwargs["num_groups"] = _find_optimal_num_groups(normalized_shape)
    elif layer in ["LayerNorm", "RMSNorm"]:
        kwargs["normalized_shape"] = normalized_shape
    elif layer != "None":
        kwargs["num_features"] = normalized_shape
    return kwargs


def check_bias_from_norm_cfg(norm_cfg: LayerCfg | None) -> bool:
    if norm_cfg is None:
        return True
    if isinstance(norm_cfg, dict):
        norm_cfg = norm_cfg["type"]
    return not norm_cfg.startswith("BatchNorm") and not norm_cfg.startswith("InstanceNorm")


def build_norm_layer(normalized_shape: int, norm_cfg: LayerCfg) -> Module:
    """Factory function for normalization layers."""
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS, f"Illegal normalization: {layer}."
    kwargs = _configure_norm_args(layer, kwargs, normalized_shape)
    return NORM_LAYERS[layer](**kwargs)


def build_norm_layer_pack_mode(normalized_shape: int, norm_cfg: LayerCfg) -> Module:
    """Factory function for normalization layers in pack mode."""
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS_PACK_MODE, f"Illegal normalization: {layer}."
    kwargs = _configure_norm_args(layer, kwargs, normalized_shape)
    return NORM_LAYERS_PACK_MODE[layer](**kwargs)


def _configure_act_args(layer: str, kwargs: dict) -> dict:
    """Configure activation args."""
    if layer == "LeakyReLU":
        if "negative_slope" not in kwargs:
            kwargs["negative_slope"] = 0.2
    return kwargs


def build_act_layer(act_cfg: LayerCfg) -> Module:
    """Factory function for activation functions."""
    layer, kwargs = parse_cfg(act_cfg)
    assert layer in ACT_LAYERS, f"Illegal activation: {layer}."
    kwargs = _configure_act_args(layer, kwargs)
    return ACT_LAYERS[layer](**kwargs)


def build_conv_layer(conv_cfg: LayerCfg) -> Module:
    """Factory function for convolution or linear layers."""
    layer, kwargs = parse_cfg(conv_cfg)
    assert layer in CONV_LAYERS, f"Illegal conv layer: {layer}."
    return CONV_LAYERS[layer](**kwargs)
