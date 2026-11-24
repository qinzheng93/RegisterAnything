from typing import Literal, overload

import torch
from torch import Tensor


@overload
def back_project(
    depth_mat: Tensor,
    intrinsics: Tensor,
    *,
    scaling_factor: float = ...,
    depth_limit: float | None = ...,
    transposed: bool = ...,
    return_mask: Literal[True],
) -> tuple[Tensor, Tensor]: ...


@overload
def back_project(
    depth_mat: Tensor,
    intrinsics: Tensor,
    *,
    scaling_factor: float = ...,
    depth_limit: float | None = ...,
    transposed: bool = ...,
    return_mask: Literal[False] = ...,
) -> Tensor: ...


def back_project(
    depth_mat: Tensor,
    intrinsics: Tensor,
    *,
    scaling_factor: float = 1000.0,
    depth_limit: float | None = None,
    transposed: bool = False,
    return_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Back project depth image to point cloud.

    Args:
        depth_mat (Tensor): the depth image with shape (B, H, W).
        intrinsics (Tensor): the intrinsic matrix with shape (B, 3, 3).
        scaling_factor (float): the depth scaling factor. Default: 1000.
        depth_limit (float, optional): ignore the pixels further than this value.
        transposed (bool): if True, the resulting point matrix is with shape (B, H, W, 3).
        return_mask (bool): if True, return a mask matrix where 0-depth points are False. Default: False.

    Returns:
        A Tensor of the point image with shape (B, 3, H, W).
        A Tensor of the mask image with shape (B, H, W).
    """
    focal_x = intrinsics[..., 0:1, 0:1]
    focal_y = intrinsics[..., 1:2, 1:2]
    center_x = intrinsics[..., 0:1, 2:3]
    center_y = intrinsics[..., 1:2, 2:3]

    batch_size, height, width = depth_mat.shape
    coords = torch.arange(height * width).view(height, width).to(depth_mat.device).unsqueeze(0).expand_as(depth_mat)
    u = coords % width  # (B, H, W)
    v = torch.div(coords, width, rounding_mode="floor")  # (B, H, W)

    z = depth_mat / scaling_factor  # (B, H, W)
    if depth_limit is not None:
        z.masked_fill_(torch.gt(z, depth_limit), 0.0)
    x = (u - center_x) * z / focal_x  # (B, H, W)
    y = (v - center_y) * z / focal_y  # (B, H, W)

    if transposed:
        points = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)
    else:
        points = torch.stack([x, y, z], dim=1)  # (B, 3, H, W)

    if not return_mask:
        return points

    masks = torch.gt(z, 0.0)

    return points, masks
