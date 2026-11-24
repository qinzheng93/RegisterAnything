import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VNLinear(nn.Module):
    """Linear layer in Vector Neurons.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize a VNLinear layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """VNLinear layer forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, D, 3, *).
        """
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]

        # reshape
        x = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)

        # project
        x_out = self.proj(x)  # (N, D, 3, X)

        # reshape
        x_out = x_out.view(batch_size, -1, 3, *other_dimensions)  # (N, D, 3, *)

        return x_out


def vn_leaky_relu(x: Tensor, d: Tensor, negative_slope: float, eps: float = 1e-9) -> Tensor:
    """Compute VNLeakyReLU.

    Args:
        x (Tensor): Input tensor with shape (N, C, 3, *).
        d (Tensor): Direction features with shape (N, C, 3, *).
        negative_slope (float): Negative slope of the leaky ReLU.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.

    Returns:
        Tensor: Output features with shape (N, C, 3, *).
    """
    dot_prod = (x * d).sum(2, keepdim=True)  # (N, C, 1, *)
    mask = torch.ge(dot_prod, 0).float()  # (N, C, 1, *)
    d_norm_sq = (d * d).sum(2, keepdim=True)  # (N, C, 1, *)
    proj = (dot_prod / (d_norm_sq + eps)) * d  # (N, C, 3, *)
    # y = mask * x + (1.0 - mask) * (x - proj)  # (N, C, 3, *)
    # x_out = negative_slope * x + (1.0 - negative_slope) * y  # (N, C, 3, *)
    x_out = x - (1.0 - negative_slope) * (1.0 - mask) * proj  # (N, C, 3, *), derived from above two lines
    return x_out


class VNLeakyReLU(nn.Module):
    """LeakyReLU layer in Vector Neurons.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(
        self, in_channels: int, share_nonlinearity: bool = False, negative_slope: float = 0.2, eps: float = 1e-9
    ):
        """Initialize a VNLeakyReLU layer.

        Args:
            in_channels (int): Number of input channels.
            share_nonlinearity (bool, optional): If True, all channels share the same direction. Defaults to False.
            negative_slope (float, optional): Negative slope of the leaky ReLU. Defaults to 0.2.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.
        """
        super().__init__()
        self.negative_slope = negative_slope
        self.eps = eps

        dir_channels = 1 if share_nonlinearity else in_channels
        self.d_proj = nn.Conv2d(in_channels, dir_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """VNLeakyReLU layer forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, C, 3, *).
        """
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]

        # reshape
        x = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)

        # dir mapping
        d = self.d_proj(x)  # (N, C, 3, X) or (N, 1, 3, X)

        # project
        x_out = vn_leaky_relu(x, d, self.negative_slope, eps=self.eps)

        # reshape
        x_out = x_out.view(batch_size, -1, 3, *other_dimensions)

        return x_out


class VNBatchNorm(nn.Module):
    """BatchNorm layer in Vector Neurons.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(self, num_features: int, eps: float = 1e-9):
        """Initialize a VNBatchNorm layer.

        Args:
            num_features (int): Number of input channels.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """VNBatchNorm layer forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, C, 3, *).
        """
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]

        # reshape
        x = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)

        # norm
        norm = torch.linalg.norm(x, dim=2, keepdim=True)  # (N, C, 1, X)
        norm_bn = self.bn(norm)  # (N, C, 1, X)
        x_out = x / (norm + self.eps) * norm_bn

        # reshape
        x_out = x_out.view(batch_size, -1, 3, *other_dimensions)  # (N, C, 3, *)

        return x


class VNL2Norm(nn.Module):
    """L2Norm layer in Vector Neurons.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(self, eps: float = 1e-9):
        """Initialize a VNL2Norm layer.

        Args:
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """VNL2Norm layer forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, C, 3, *).
        """
        x_out = F.normalize(x + self.eps, dim=2, p=2)
        return x_out


class VNInstanceNorm(nn.Module):
    """LayerNorm layer in Vector Neurons.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(self, num_features: int):
        """Initialize a VNInstanceNorm layer.

        Args:
            num_features (int): Number of input channels.
        """
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """VNInstanceNorm layer forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, C, 3, *).
        """
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]

        # reshape
        x = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)

        # norm
        x_out = self.norm(x)  # (N, C, 3, X)

        # reshape
        x_out = x_out.view(batch_size, -1, 3, *other_dimensions)  # (N, C, 3, *)

        return x_out


class VNLayerNorm(nn.Module):
    """LayerNorm layer in Vector Neurons.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(self, num_features: int):
        """Initialize a VNLayerNorm layer.

        Args:
            num_features (int): Number of input channels.
        """
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """VNLayerNorm layer forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, C, 3, *).
        """
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[2:]

        # reshape
        x = x.view(batch_size, in_channels, -1).transpose(1, 2)  # (N, X, C)

        # norm
        x_out = self.norm(x)  # (N, X, C)

        # reshape
        x_out = x_out.transpose(1, 2).reshape(batch_size, -1, *other_dimensions)  # (N, C, *)

        return x_out


class VNLinearBlock(nn.Module):
    """Linear block in Vector Neurons.

    This block contains a VNLinear, a VN normalization, and a VNLeakyReLU.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        dir_from_input: bool = True,
        eps: float = 1e-9,
    ):
        """Initialize a VNLinearBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batchnorm (bool, optional): If True, use batch normalization, and use L2-norm otherwise. Defaults to False.
            share_nonlinearity (bool, optional): If True, all channels share the same direction in LeakyReLU. Defaults to False.
            negative_slope (float, optional): Negative slope of in LeakyReLU. Defaults to 0.2.
            dir_from_input (bool, optional): If True, the direction is computed from the input. Defaults to True.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.
        """
        super().__init__()
        self.negative_slope = negative_slope
        self.dir_from_input = dir_from_input
        self.eps = eps

        self.f_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        if use_batchnorm:
            self.norm = VNBatchNorm(out_channels, eps=eps)
        else:
            self.norm = VNL2Norm()

        dir_channels = 1 if share_nonlinearity else out_channels
        self.d_proj = nn.Conv2d(in_channels, dir_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """VNLinearBlock forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, D, 3, *).
        """
        # reshape
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]
        x = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)

        # Linear
        p = self.f_proj(x)  # (N, D, 3, X)

        # BatchNorm
        p = self.norm(p)

        # LeakyReLU
        if self.dir_from_input:
            d = self.d_proj(x)  # (N, D, 3, X) or (N, 1, 3, X)
        else:
            d = self.d_proj(p)  # (N, D, 3, X) or (N, 1, 3, X)
        x_out = vn_leaky_relu(p, d, self.negative_slope, eps=self.eps)  # (N, D, 3, X)

        # reshape
        x_out = x_out.view(batch_size, -1, 3, *other_dimensions)  # (N, D, 3, *)

        return x_out


class VNMaxPool(nn.Module):
    """Vector-Neuron Max Pooling along the last axis.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(self, in_channels: int):
        """Vector-Neuron Max Pooling along the last axis.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """VNMaxPool forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            Tensor: Output tensor with shape (N, C, 3, *).
        """
        # reshape
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]
        x_reshape = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)

        # projection
        d = self.proj(x_reshape)  # (N, C, 3, X)

        # reshape
        d = d.view(batch_size, -1, 3, *other_dimensions)  # (N, C, 3, *)

        # maxpool
        dotprod = (x * d).sum(2, keepdims=True)  # (N, C, 1, *)
        max_indices = dotprod.max(dim=-1, keepdim=True)[1]  # (N, C, 1, *, 1)
        x_max = x.gather(-1, max_indices)  # (N, C, 3, *, 1)

        return x_max


class VNStdFeature(nn.Module):
    """Vector-Neuron Invariant Feature Layer.

    Paper: https://arxiv.org/abs/2104.12229.

    Ported from https://github.com/FlyingGiraffe/vnn.
    """

    def __init__(
        self,
        in_channels: int,
        normalize_frame: bool = False,
        use_batchnorm: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        dir_from_input: bool = True,
        eps: float = 1e-9,
    ):
        """Initialize a VNStdFeature Layer.

        Args:
            in_channels (int): Number of input channels.
            dim (int, optional): Dimension of the input tensor. Defaults to 4.
            normalize_frame (bool, optional): If True, normalize the frame. Defaults to False.
            use_batchnorm (bool, optional): If True, use batch normalization, and use L2-norm otherwise. Defaults to False.
            share_nonlinearity (bool, optional): If True, all channels share the same direction in LeakyReLU. Defaults to False.
            negative_slope (float, optional): Negative slope of in LeakyReLU. Defaults to 0.2.
            dir_from_input (bool, optional): If True, the direction is computed from the input. Defaults to True.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-9.
        """
        super().__init__()
        self.normalize_frame = normalize_frame
        self.eps = eps

        self.proj1 = VNLinearBlock(
            in_channels,
            in_channels // 2,
            use_batchnorm=use_batchnorm,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            dir_from_input=dir_from_input,
            eps=eps,
        )
        self.proj2 = VNLinearBlock(
            in_channels // 2,
            in_channels // 4,
            use_batchnorm=use_batchnorm,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            dir_from_input=dir_from_input,
            eps=eps,
        )
        f_channels = 2 if normalize_frame else 3
        self.proj3 = VNLinear(in_channels // 4, f_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """VNStdFeature forward propagation.

        Args:
            x (Tensor): Input tensor with shape (N, C, 3, *).

        Returns:
            tuple[Tensor, Tensor]: Output tensor with shape (N, C, 3, *) and frame tensor with shape (N, 2 or 3, 3, *).
        """
        # vn blocks
        z0 = self.proj1(x)  # (N, C/2, 3, *)
        z0 = self.proj2(z0)  # (N, C/4, 3, *)
        z0 = self.proj3(z0)  # (N, 2 or 3, 3, *)

        # compute normalized frame
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]  # (N, 3, *)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))  # (N, 1, *)
            u1 = v1 / (v1_norm + self.eps)  # (N, 3, *)
            v2 = z0[:, 1, :]  # (N, 3, *)
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1  # (N, 3, *)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))  # (B, 1, *)
            u2 = v2 / (v2_norm + self.eps)  # (N, 3, *)

            # compute the cross product of the two output vectors
            u3 = torch.linalg.cross(u1, u2, dim=1)  # (B, 3, *)
            z0 = torch.stack([u1, u2, u3], dim=1)  # (B, 3, 3, *)

        # For z0, the vn dim is switched to dim-1
        z0 = z0.transpose(1, 2)  # (B, 3, 3, *)

        # record shape
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        other_dimensions = x.shape[3:]

        # reshape
        x_reshape = x.view(batch_size, in_channels, 3, -1)  # (N, C, 3, X)
        z0_reshape = z0.view(batch_size, 3, 3, -1)  # (N, 3, 3, X)

        # compute std
        x_std = torch.einsum("bijm,bjkm->bikm", x_reshape, z0_reshape)  # (N, C, 3, X)

        # reshape
        x_std = x_std.view(batch_size, in_channels, 3, *other_dimensions)  # (N, C, 3, *)

        return x_std, z0
