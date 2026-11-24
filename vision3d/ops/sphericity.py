import torch
from torch import Tensor

from .masked_ops import masked_mean


def point_cloud_analysis(points: Tensor, masks: Tensor | None = None, eps: float = 1e-6) -> tuple[Tensor, ...]:
    """Computes the sphericity of a point cloud using PCA.

    Args:
        points (Tensor): Input point cloud with shape (B, N, 3).
        masks (Tensor | None): Input mask with shape (B, N). Default: None.

    Returns:
        A float of the sphericity of the point cloud.
        A bool of whether the point cloud is aligned with the global z-axis.
        A PCA object of the estimated transformation.
    """
    if masks is not None:
        num_points = masks.sum(1)  # (B,)
        centroids = masked_mean(points, masks)  # (B, 3)
        points = points - centroids.unsqueeze(1)  # (B, N, 3)
        cov_matrix = torch.matmul(points.transpose(1, 2), points)  # (B, 3, 3)
        cov_matrix = cov_matrix / (num_points.view(-1, 1, 1) + eps)  # (B, 3, 3)
    else:
        num_points = points.shape[1]
        centroids = points.mean(1)  # (B, 3)
        points = points - centroids.unsqueeze(1)  # (B, N, 3)
        cov_matrix = torch.matmul(points.transpose(1, 2), points)  # (B, 3, 3)
        cov_matrix = cov_matrix / float(num_points)  # (B, 3, 3)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # (B, 3), (B, 3, 3)

    # lambda1 > lambda2 > lambda3
    lambda3 = eigenvalues[:, 0]  # (B,)
    lambda2 = eigenvalues[:, 1]  # (B,)
    lambda1 = eigenvalues[:, 2]  # (B,)

    linearity = (lambda1 - lambda2) / lambda1
    planarity = (lambda2 - lambda3) / lambda1
    sphericity = lambda3 / lambda1
    anisotropy = (lambda1 - lambda3) / lambda1

    return linearity, planarity, sphericity, anisotropy
