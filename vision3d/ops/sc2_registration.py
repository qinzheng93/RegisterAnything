import torch
from torch import Tensor

from .eigenvector import leading_eigenvector
from .spatial_consistency import second_order_spatial_consistency
from .weighted_procrustes import weighted_procrustes


def sc2_registration(
    src_points: Tensor,
    tgt_points: Tensor,
    sigma: float,
    masks: Tensor | None = None,
    k: int = 10,
    tight_metric: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """SC^2 Registration algorithm.

    Proposed in *SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration*, CVPR 2022.

    Args:
        src_points (Tensor): The corresponding points in the source point cloud with shape (B, N, 3) or (N, 3)
        tgt_points (Tensor): The corresponding points in the target point cloud with shape (B, N, 3) or (N, 3)
        sigma (float): The distance threshold for spatial consistency.
        masks (Tensor, optional): The masks of the correspondences with shape (B, N) or (N,). True is valid. Default: None.
        k (int, optional): The number of correspondences to select. Default: 10.
        tight_metric (bool, optional): Whether to use tight metric. Default: False.
        eps (float, optional): The epsilon value for numerical stability. Default: 1e-6.

    Returns:
        Tensor: The transform matrix with shape (B, 4, 4) or (4, 4)
    """
    squeeze_first = False
    if src_points.dim() == 2:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if masks is not None:
            masks = masks.float().unsqueeze(0)
        squeeze_first = True

    batch_size = src_points.shape[0]
    batch_index = torch.arange(batch_size).cuda()  # (B,)

    # compute spatial consistency matrix
    sc1_mat, sc2_mat = second_order_spatial_consistency(
        src_points, tgt_points, sigma, tight_metric=tight_metric
    )  # (B, N, N), (B, N, N)

    if masks is not None:
        sc1_mat = sc1_mat * masks.unsqueeze(-1) * masks.unsqueeze(-2)  # (B, N, N)
        sc2_mat = sc2_mat * masks.unsqueeze(-1) * masks.unsqueeze(-2)  # (B, N, N)

    # compute leading eigenvector as correspondence scores
    digonal_masks = 1.0 - torch.eye(sc1_mat.shape[-1]).unsqueeze(0).cuda()  # (N, N)
    sc1_mat = sc1_mat * digonal_masks  # (B, N, N)
    weights = leading_eigenvector(sc1_mat, method="power")  # (B, N)
    if masks is not None:
        weights = weights * masks

    # best correspondence as seed for each batch
    best_index = torch.argmax(weights, dim=-1)  # (B,)

    # select k correspondences with highest SC^2 scores
    k = min(k, src_points.shape[1])
    corr_scores = sc2_mat[batch_index, best_index]  # (B, N)
    sel_corr_indices = corr_scores.topk(k=k, dim=-1, largest=True)[1]  # (B, k)
    sel_weights = torch.gather(weights, dim=1, index=sel_corr_indices)  # (B, k)
    sel_corr_indices = sel_corr_indices.unsqueeze(-1).expand(-1, -1, 3)  # (B, k, 3)
    sel_src_points = torch.gather(src_points, dim=1, index=sel_corr_indices)  # (B, k, 3)
    sel_tgt_points = torch.gather(tgt_points, dim=1, index=sel_corr_indices)  # (B, k, 3)

    # solve transformation
    transform = weighted_procrustes(sel_src_points, sel_tgt_points, weights=sel_weights)

    if squeeze_first:
        transform = transform.squeeze(0)

    return transform
