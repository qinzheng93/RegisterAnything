import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA

from .masked_ops import masked_mean


def point_pca_with_sklearn(points: ndarray) -> tuple[float, float, float, float, ndarray]:
    """Computes the linearity, planarity, sphericity, anisotropy, and principal axes of a point cloud using sklearn PCA.

    Args:
        points (ndarray): Input point cloud with shape (N, 3).

    Returns:
        A float of the linearity of the point cloud.
        A float of the planarity of the point cloud.
        A float of the sphericity of the point cloud.
        A float of the anisotropy of the point cloud.
        A ndarray of the priciple axes of the point cloud.
    """
    pca = PCA(n_components=3)
    pca.fit(points)

    # 1. eigenvalue-based geometric features
    lambda1, lambda2, lambda3 = pca.explained_variance_

    linearity = (lambda1 - lambda2) / lambda1
    planarity = (lambda2 - lambda3) / lambda1
    sphericity = lambda3 / lambda1
    anisotropy = (lambda1 - lambda3) / lambda1

    # 2. eigenvector-based orthogonal axes
    eigenvectors = pca.components_  # (3, 3)
    axes = eigenvectors / np.linalg.norm(eigenvectors, axis=-1, keepdims=True)

    return linearity, planarity, sphericity, anisotropy, axes


def point_pca(
    points: ndarray, masks: ndarray | None = None, descending: bool = True, eps: float = 1e-9
) -> tuple[ndarray, ...]:
    """Computes the eigenvalues and the eigenvectors of a point cloud using PCA.

    Args:
        points (ndarray): Input point cloud with shape (B, N, 3) or (N, 3).
        masks (ndarray | None): Input mask with shape (B, N) or (N,). Default: None.
        descending (bool): Whether to sort the eigenvalues in descending order. Default: True.
        eps (float): A small constant to avoid numerical instability. Default: 1e-9.

    Returns:
        A ndarray of the eigenvalues of the point cloud sorted in descending order with shape (..., 3).
        A ndarray of the corresponding eigenvectors of the point cloud with shape (..., 3, 3).
    """
    squeeze_first = False
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
        if masks is not None:
            masks = np.expand_dims(masks, axis=0)
        squeeze_first = True

    if masks is None:
        masks = np.ones_like(points[..., 0], dtype=bool)

    centroids = masked_mean(points, masks, keepdims=True)  # (B, 1, 3)
    num_points = masks.sum(1).reshape(-1, 1, 1)  # (B, 1, 1)
    points = points - centroids  # (B, N, 3)
    cov_matrix = points.transpose(0, 2, 1) @ (masks[..., np.newaxis] * points)  # (B, 3, 3)
    cov_matrix = cov_matrix / (num_points - 1 + eps)  # (B, 3, 3)

    # decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # (B, 3), (B, 3, 3)
    eigenvalues = eigenvalues.clip(min=eps)  # eusure non-negative

    # sort in descending order
    if descending:
        eigenvalues = eigenvalues[..., ::-1]  # (B, 3)
        eigenvectors = eigenvectors[..., ::-1]  # (B, 3, 3)

    # transpose eigenvectors
    eigenvectors = eigenvectors.transpose(0, 2, 1)  # (B, 3, 3)

    # handle squeeze dim
    if squeeze_first:
        eigenvalues = eigenvalues.squeeze(0)
        eigenvectors = eigenvectors.squeeze(0)

    return eigenvalues, eigenvectors


def compute_point_pca_features(eigenvalues: ndarray) -> tuple[ndarray, ...]:
    """Computes the linearity, planarity, sphericity, and anisotropy of a point cloud using PCA eigenvalues.

    Args:
        eigenvalues (ndarray): Input eigenvalues with shape (..., 3), sorted in descending order.

    Returns:
        A ndarray of the linearity of the point cloud.
        A ndarray of the planarity of the point cloud.
        A ndarray of the sphericity of the point cloud.
        A ndarray of the anisotropy of the point cloud.
    """
    # 1. eigenvalue-based geometric features
    # lambda1 > lambda2 > lambda3
    lambda1 = eigenvalues[..., 0]  # (...,)
    lambda2 = eigenvalues[..., 1]  # (...,)
    lambda3 = eigenvalues[..., 2]  # (...,)

    lambda_sum = lambda1 + lambda2 + lambda3

    linearity = (lambda1 - lambda2) / lambda1
    planarity = (lambda2 - lambda3) / lambda1
    sphericity = lambda3 / lambda1

    anisotropy = (lambda1 - lambda3) / lambda1

    omnivariance = (lambda1 * lambda2 * lambda3) ** (1.0 / 3.0)
    eigenentropy = -np.sum(eigenvalues * np.log(eigenvalues), axis=-1)
    variation = lambda3 / lambda_sum

    return linearity, planarity, sphericity, anisotropy, omnivariance, eigenentropy, variation
