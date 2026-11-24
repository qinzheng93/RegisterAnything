import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA


def compute_sphericity_with_pca(
    points: ndarray, z_alignment_threshold: float = 0.98, keep_ratio: float | None = None
) -> tuple[float, bool, PCA]:
    """Computes the sphericity of a point cloud using PCA.

    Args:
        points (ndarray): Input point cloud with shape (N, 3).
        z_alignment_threshold (float): Threshold for z-axis alignment. Defaults to 0.98.
        keep_ratio (float | None): Ratio of points to keep. Defaults to None.

    Returns:
        A float of the sphericity of the point cloud.
        A bool of whether the point cloud is aligned with the global z-axis.
        A PCA object of the estimated transformation.
    """
    if keep_ratio is not None:
        # sample 10% of the points
        num_points = points.shape[0]
        sample_size = int(num_points * keep_ratio)
        indices = np.random.choice(num_points, size=sample_size, replace=False)
        points = points[indices]

    # PCA
    pca = PCA(n_components=3)
    pca.fit(points)

    # 1. eigenvalue-based sphericity
    eigenvalues = pca.explained_variance_
    lambda1, _, lambda3 = sorted(eigenvalues, reverse=True)
    sphericity = lambda3 / lambda1

    # 2. eigenvector-based principle axes
    z_axis_candidate = pca.components_[-1]
    z_axis_candidate = z_axis_candidate / np.linalg.norm(z_axis_candidate)
    z_alignment_score = np.abs(np.dot(z_axis_candidate, np.array([0, 0, 1])))
    is_aligned = z_alignment_score > z_alignment_threshold

    return sphericity, is_aligned, pca


def sphericity_based_voxel_analyze(
    src_points: ndarray,
    tgt_points: ndarray,
    sphericity_threshold: float = 0.05,
    z_alignment_threshold: float = 0.98,
    keep_ratio: float | None = 0.1,
):
    """Estimates voxel size and sphericity, and checks if both source and target point clouds
    are aligned with the global z-axis using PCA.

    Args:
        src_points (ndarray): Source point cloud with shape (N, 3).
        tgt_points (ndarray): Target point cloud with shape (M, 3).
        sphericity_threshold (float): Threshold for sphericity. Defaults to 0.05.
        z_alignment_threshold (float): Threshold for z-axis alignment. Defaults to 0.98.
        keep_ratio (float | None): Ratio of points to keep. Defaults to 0.1.

    Returns:
        A float of the estimated voxel size.
        A float of the sphericity of the denser point cloud.
        A bool of whether both point clouds are aligned with the global z-axis.
    """
    # PCA for both src and tgt
    src_sphericity, src_is_aligned, src_pca = compute_sphericity_with_pca(
        src_points, z_alignment_threshold=z_alignment_threshold, keep_ratio=keep_ratio
    )
    tgt_sphericity, tgt_is_aligned, tgt_pca = compute_sphericity_with_pca(
        tgt_points, z_alignment_threshold=z_alignment_threshold, keep_ratio=keep_ratio
    )

    # Use denser point cloud for voxel size estimation
    if src_points.shape[0] > tgt_points.shape[0]:
        ref_points = src_points
        sphericity = src_sphericity
        pca = src_pca
    else:
        ref_points = tgt_points
        sphericity = tgt_sphericity
        pca = tgt_pca

    # estimate voxel size
    tsfm_points = pca.transform(ref_points)
    z_range = tsfm_points[:, 2].max() - tsfm_points[:, 2].min()
    alpha = 1.0 if sphericity < sphericity_threshold else 1.5
    voxel_size = np.sqrt(z_range) / 100 * alpha
    voxel_size = max(voxel_size, 0.001)

    z_src = src_pca.components_[-1]
    z_tgt = tgt_pca.components_[-1]

    z_src /= np.linalg.norm(z_src)
    z_tgt /= np.linalg.norm(z_tgt)

    dot_z = np.dot(z_src, z_tgt)
    same_direction = dot_z > 0.96

    is_aligned_to_global_z = src_is_aligned and tgt_is_aligned and same_direction

    return round(voxel_size, 4), sphericity, is_aligned_to_global_z
