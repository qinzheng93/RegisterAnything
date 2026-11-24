from typing import Literal, overload

import torch
from torch import Tensor

from .pairwise_distance import pairwise_distance


def spatial_consistency(src_corr_points: Tensor, tgt_corr_points: Tensor, sigma: float) -> Tensor:
    """Compute spatial consistency.

    SC_{i,j} = max(1 - d_{i,j}^2 / sigma^2, 0)
    d_{i,j} = abs(|p_i - p_j| - |q_i - q_j|)

    Args:
        src_corr_points (Tensor): The correspondence points in the source point cloud with shape (*, N, 3).
        tgt_corr_points (Tensor): The correspondence points in the source point cloud with shape (*, N, 3).
        sigma (float): The sensitivity factor.

    Returns:
        Tensor: The spatial consistency between the correspondences with shape (*, N, N).
    """
    src_dist_mat = pairwise_distance(src_corr_points, src_corr_points, squared=False)  # (*, N, N)
    tgt_dist_mat = pairwise_distance(tgt_corr_points, tgt_corr_points, squared=False)  # (*, N, N)
    delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)
    sc1_mat = torch.relu(1.0 - delta_mat**2 / sigma**2)
    return sc1_mat


def cross_spatial_consistency(
    q_src_corr_points: Tensor,
    q_tgt_corr_points: Tensor,
    s_src_corr_points: Tensor,
    s_tgt_corr_points: Tensor,
    sigma: float,
) -> Tensor:
    """Compute cross spatial consistency between two set of correspondences.

    SC_{i,j} = max(1 - d_{i,j}^2 / sigma^2, 0)
    d_{i,j} = abs(|p_i - p_j| - |q_i - q_j|)

    Args:
        q_src_corr_points (Tensor): the query correspondence points in the source point cloud (*, N, 3).
        q_tgt_corr_points (Tensor): the query correspondence points in the target point cloud (*, N, 3).
        s_src_corr_points (Tensor): the support correspondence points in the source point cloud (*, M, 3).
        s_tgt_corr_points (Tensor): the support correspondence points in the target point cloud (*, M, 3).
        sigma (float): The sensitivity factor.

    Returns:
        Tensor: The spatial consistency between the correspondences with shape (*, N, M).
    """
    src_dist_mat = pairwise_distance(q_src_corr_points, s_src_corr_points, squared=False)  # (*, N, N)
    tgt_dist_mat = pairwise_distance(q_tgt_corr_points, s_tgt_corr_points, squared=False)  # (*, N, N)
    delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)
    sc1_mat = torch.relu(1.0 - delta_mat**2 / sigma**2)
    return sc1_mat


@overload
def second_order_spatial_consistency(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: Literal[False] = False,
    return_soft_sc2: Literal[False] = False,
) -> Tensor: ...


@overload
def second_order_spatial_consistency(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: Literal[True],
    return_soft_sc2: bool = False,
) -> tuple[Tensor, ...]: ...


@overload
def second_order_spatial_consistency(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: bool = False,
    return_soft_sc2: Literal[True],
) -> tuple[Tensor, ...]: ...


def second_order_spatial_consistency(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: bool = False,
    return_soft_sc2: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    """Compute 2-order spatial consistency.

    d_{i,j} = abs(|p_i - p_j| - |q_i - q_j|)
    SC_{i,j} = max(1 - d_{i,j}^2 / sigma^2, 0)
    COMP_{i,j} = d_{i, j} < sigma
    SC2_{i,j} = COMP_{i,j} * (sum_{k=1}^N COMP_{i,k} * COMP_{k,j})

    Args:
        src_corr_points (Tensor): The correspondence points in the source point cloud with shape (*, N, 3).
        tgt_corr_points (Tensor): The correspondence points in the target point cloud with shape (*, N, 3).
        sigma (float): The distance threshold.
        return_sc1 (bool, optional): Whether to return 1-order spatial consistency. Default: False.
        return_soft_sc2 (bool, optional): Whether to return soft 2-order spatial consistency. Default: False.

    Returns:
        Tensor: The 2-order spatial consistency between the correspondences with shape (*, N, N).
        Tensor (optional): The 1-order spatial consistency between the correspondences with shape (*, N, N).
        Tensor (optional): The soft 2-order spatial consistency between the correspondences with shape (*, N, N).
    """

    # pairwise distance
    src_dist_mat = pairwise_distance(src_corr_points, src_corr_points, squared=False)  # (*, N, N)
    tgt_dist_mat = pairwise_distance(tgt_corr_points, tgt_corr_points, squared=False)  # (*, N, N)
    delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)  # (*, N, N)

    # hard 1-order SC
    hard_sc1_mat = torch.lt(delta_mat, distance_threshold).float()  # (*, N, N)

    # 2-order SC
    sc2_mat = hard_sc1_mat * torch.matmul(hard_sc1_mat, hard_sc1_mat)  # (*, N, N)

    if not return_sc1 and not return_soft_sc2:
        return sc2_mat

    output_list = [sc2_mat]

    # 1-order SC
    sc1_mat = torch.relu(1.0 - delta_mat**2 / distance_threshold**2)

    if return_sc1:
        output_list.append(sc1_mat)

    # soft 2-order SC
    if return_soft_sc2:
        soft_sc2_mat = sc1_mat * torch.matmul(sc1_mat, sc1_mat)  # (*, N, N)
        output_list.append(soft_sc2_mat)

    return tuple(output_list)


@overload
def cross_second_order_spatial_consistency(
    q_src_corr_points: Tensor,
    q_tgt_corr_points: Tensor,
    s_src_corr_points: Tensor,
    s_tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: Literal[False] = False,
    return_soft_sc2: Literal[False] = False,
) -> Tensor: ...


@overload
def cross_second_order_spatial_consistency(
    q_src_corr_points: Tensor,
    q_tgt_corr_points: Tensor,
    s_src_corr_points: Tensor,
    s_tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: Literal[True],
    return_soft_sc2: bool = False,
) -> Tensor: ...


@overload
def cross_second_order_spatial_consistency(
    q_src_corr_points: Tensor,
    q_tgt_corr_points: Tensor,
    s_src_corr_points: Tensor,
    s_tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: bool = False,
    return_soft_sc2: Literal[True],
) -> Tensor: ...


def cross_second_order_spatial_consistency(
    q_src_corr_points: Tensor,
    q_tgt_corr_points: Tensor,
    s_src_corr_points: Tensor,
    s_tgt_corr_points: Tensor,
    distance_threshold: float,
    *,
    return_sc1: bool = False,
    return_soft_sc2: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    """Compute cross 2-order spatial consistency between two set of correspondences.

    d_{i,j} = abs(|p_i - p_j| - |q_i - q_j|)
    SC_{i,j} = max(1 - d_{i,j}^2 / sigma^2, 0)
    COMP_{i,j} = d_{i, j} < sigma
    SC2_{i,j} = COMP_{i,j} * (sum_{k=1}^N COMP_{i,k} * COMP_{k,j})

    Args:
        q_src_corr_points (Tensor): The query correspondences in the source point cloud with shape of (*, N, 3).
        q_tgt_corr_points (Tensor): The query correspondences in the target point cloud with shape of (*, N, 3).
        q_src_corr_points (Tensor): The support correspondences in the source point cloud with shape of (*, M, 3).
        q_tgt_corr_points (Tensor): The support correspondences in the target point cloud with shape of (*, M, 3).
        sigma (float): The distance threshold.
        return_sc1 (bool, optional): Whether to return 1-order spatial consistency. Default: False.
        return_soft_sc2 (bool, optional): Whether to return soft 2-order spatial consistency. Default: False.

    Returns:
        Tensor: The 2-order spatial consistency between the correspondences with shape (*, N, M).
        Tensor (optional): The 1-order spatial consistency between the correspondences with shape (*, N, M).
        Tensor (optional): The soft 2-order spatial consistency between the correspondences with shape (*, N, M).
    """
    # cross delta mat
    src_dist_mat = pairwise_distance(q_src_corr_points, s_src_corr_points, squared=False)  # (*, N, M)
    tgt_dist_mat = pairwise_distance(q_tgt_corr_points, s_tgt_corr_points, squared=False)  # (*, N, M)
    delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)  # (*, N, M)

    # support delta mat
    s_src_dist_mat = pairwise_distance(s_src_corr_points, s_src_corr_points, squared=False)  # (*, M, M)
    s_tgt_dist_mat = pairwise_distance(s_tgt_corr_points, s_tgt_corr_points, squared=False)  # (*, M, M)
    s_delta_mat = torch.abs(s_src_dist_mat - s_tgt_dist_mat)  # (*, M, M)

    # hard 1-order SC
    hard_sc1_mat = torch.lt(delta_mat, distance_threshold).float()  # (*, N, M)
    s_hard_sc1_mat = torch.lt(s_delta_mat, distance_threshold).float()  # (*, M, M)

    # 2-order SC
    sc2_mat = hard_sc1_mat * torch.matmul(hard_sc1_mat, s_hard_sc1_mat)  # (*, N, M)

    if not return_sc1 and not return_soft_sc2:
        return sc2_mat

    output_list = [sc2_mat]

    # 1-order SC
    sc1_mat = torch.relu(1.0 - delta_mat**2 / distance_threshold**2)  # (*, N, M)

    if return_sc1:
        output_list.append(sc1_mat)

    # soft 2-order SC
    if return_soft_sc2:
        s_sc1_mat = torch.relu(1.0 - s_delta_mat**2 / distance_threshold**2)  # (*, M, M)
        soft_sc2_mat = sc1_mat * torch.matmul(sc1_mat, s_sc1_mat)  # (*, N, M)
        output_list.append(soft_sc2_mat)

    return tuple(output_list)
