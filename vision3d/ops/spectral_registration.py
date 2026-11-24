import torch
from torch import Tensor

from .eigenvector import leading_eigenvector
from .spatial_consistency import spatial_consistency
from .weighted_procrustes import weighted_procrustes


def spectral_registration(
    src_points: Tensor, tgt_points: Tensor, sigma: float, masks: Tensor | None = None, eps: float = 1e-6
) -> Tensor:
    """Spectral consistency algorithm for point cloud registration.

    Proposed in *A spectral technique for correspondence problems using pairwise constraints*, ICCV 2005.

    Args:
        src_points (Tensor): The corresponding points in the source point cloud with shape (B, N, 3) or (N, 3)
        tgt_points (Tensor): The corresponding points in the target point cloud with shape (B, N, 3) or (N, 3)
        sigma (float): The sensitivity factor.
        masks (Tensor, optional): The masks of the correspondences with shape (B, N) or (N,). True is valid. Default: None.
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

    # compute spatial consistency matrix
    sc1_mat = spatial_consistency(src_points, tgt_points, sigma)  # (B, N, N)

    if masks is not None:
        sc1_mat = sc1_mat * masks.unsqueeze(-1) * masks.unsqueeze(-2)  # (B, N, N)

    # compute leading eigenvector as correspondence scores
    digonal_masks = 1.0 - torch.eye(sc1_mat.shape[-1]).unsqueeze(0).cuda()  # (N, N)
    sc1_mat = sc1_mat * digonal_masks  # (B, N, N)
    weights = leading_eigenvector(sc1_mat, method="power")  # (B, N)
    if masks is not None:
        weights = weights * masks

    # solve transformation
    transform = weighted_procrustes(src_points, tgt_points, weights=weights)

    if squeeze_first:
        transform = transform.squeeze(0)

    return transform
