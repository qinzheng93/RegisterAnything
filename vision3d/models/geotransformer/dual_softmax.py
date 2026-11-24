import torch
import torch.nn as nn
from torch import Tensor


class DualSoftmaxWithConfidence(nn.Module):
    """Dual Softmax with Feature and Confidence Projection.

    Described in https://arxiv.org/abs/2407.10142.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inf: float = 1e12,
    ):
        """Initialize the ProjectedDualSoftmax layer.

        Args:
            input_dim (int): input feature dimension.
            output_dim (int): output feature dimension.
        """
        super().__init__()
        self.f_proj = nn.Linear(input_dim, output_dim)
        self.s_proj = nn.Linear(input_dim, 1)
        self.inf = inf

    def forward(
        self,
        src_knn_feats: Tensor,
        tgt_knn_feats: Tensor,
        src_knn_masks: Tensor | None = None,
        tgt_knn_masks: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """ProjectedDualSoftmax layer forward propagation.

        Args:
            src_knn_feats (Tensor): The neighbor features in the source point cloud with shape (B, K, C).
            tgt_knn_feats (Tensor): The neighbor features in the target point cloud with shape (B, K, C).
            src_knn_masks (Tensor, optional): The neighbor masks in the source point cloud with shape of (B, K). If False, the neighbor is invalid. Default: None.
            tgt_knn_masks (Tensor, optional): The neighbor masks in the target point cloud with shape of (B, K). If False, the neighbor is invalid. Default: None.

        Returns:
            Tensor: The projected neighbor features in the source point cloud with shape (B, K, C).
            Tensor: The projected neighbor features in the target point cloud with shape (B, K, C).
            Tensor: The projected neighbor scores in the source point cloud with shape (B, K).
            Tensor: The projected neighbor scores in the target point cloud with shape (B, K).
            Tensor: The matching scores with shape (B, K, K).
        """
        src_knn_saliencies = torch.sigmoid(self.s_proj(src_knn_feats)).squeeze(-1)  # (B, K)
        tgt_knn_saliencies = torch.sigmoid(self.s_proj(tgt_knn_feats)).squeeze(-1)  # (B, K)

        src_knn_feats = self.f_proj(src_knn_feats)  # (B, K, C)
        tgt_knn_feats = self.f_proj(tgt_knn_feats)  # (B, K, C)

        score_mat = torch.einsum("bnc,bmc->bnm", src_knn_feats, tgt_knn_feats)  # (B, K, K)
        score_mat = score_mat / src_knn_feats.shape[-1] ** 0.5

        # NOTE: process src and tgt masks separately to avoid softmax on all -inf rows or columns.
        if src_knn_masks is not None:
            src_knn_weights = torch.where(src_knn_masks, 0, -self.inf).unsqueeze(-1)  # (B, K, 1)
            src_score_mat = torch.softmax(score_mat + src_knn_weights, dim=-1)  # (B, K, K)
        else:
            src_score_mat = torch.softmax(score_mat, dim=-1)  # (B, K, K)

        if tgt_knn_masks is not None:
            tgt_knn_weights = torch.where(tgt_knn_masks, 0, -self.inf).unsqueeze(-2)  # (B, 1, K)
            tgt_score_mat = torch.softmax(score_mat + tgt_knn_weights, dim=-2)  # (B, K, K)
        else:
            tgt_score_mat = torch.softmax(score_mat, dim=-2)  # (B, K, K)

        dual_saliency_mat = src_knn_saliencies.unsqueeze(-1) * tgt_knn_saliencies.unsqueeze(-2)
        matchability_mat = dual_saliency_mat * src_score_mat * tgt_score_mat

        return src_knn_feats, tgt_knn_feats, src_knn_saliencies, tgt_knn_saliencies, matchability_mat
