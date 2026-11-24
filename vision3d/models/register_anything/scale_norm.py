import numpy as np
from numpy import ndarray

from vision3d.array_ops.furthest_point_sample import furthest_point_sample
from vision3d.array_ops.point_cloud_utils import get_bounding_box_diagonal
from vision3d.array_ops.point_pca import compute_point_pca_features, point_pca
from vision3d.array_ops.radius_search import radius_search_pack_mode
from vision3d.utils.open3d import voxel_down_sample


def compute_point_features(src_seeds, tgt_seeds, src_points, tgt_points, radius, max_neighbors, voxel_size):
    seeds = np.concatenate([src_seeds, tgt_seeds], axis=0)
    points = np.concatenate([src_points, tgt_points], axis=0)
    q_lengths = np.asarray([src_seeds.shape[0], tgt_seeds.shape[0]])
    s_lengths = np.asarray([src_points.shape[0], tgt_points.shape[0]])

    knn_indices = radius_search_pack_mode(seeds, points, q_lengths, s_lengths, radius, max_neighbors)
    padded_points = np.concatenate([points, np.zeros_like(points[:1])], axis=0)
    knn_points = padded_points[knn_indices]
    knn_masks = knn_indices < points.shape[0]
    knn_points = knn_points / voxel_size

    eigenvalues, eigenvectors = point_pca(knn_points, masks=knn_masks)
    lin, pla, sph, ani, var, ent, suf = compute_point_pca_features(eigenvalues)
    com = eigenvectors / np.linalg.norm(eigenvectors, axis=-1, keepdims=True)

    return lin, pla, sph, ani, var, ent, suf, com


def update_voxel_size(
    src_points: ndarray,
    tgt_points: ndarray,
    voxel_size: float,
    abs_tol: float = 0.1,
    rel_tol: float = 0.15,
    dir_tol: float = 0.2,
    max_attempts: int = 5,
    num_seeds: int = 1024,
    max_neighbors: int = 1024,
    radius_factor: float = 8,
    alpha: float = 0.05,
) -> float:
    best_voxel_size = voxel_size
    min_voxel_size = voxel_size * 0.8
    max_voxel_size = voxel_size * 1.2

    # compute radius
    radius = voxel_size * radius_factor

    for _ in range(max_attempts):
        # downsample points
        src_ds_points = voxel_down_sample(src_points, voxel_size)
        tgt_ds_points = voxel_down_sample(tgt_points, voxel_size)

        # sample seeds
        src_num_seeds = min(num_seeds, src_ds_points.shape[0])
        tgt_num_seeds = min(num_seeds, tgt_ds_points.shape[0])
        src_indices = furthest_point_sample(src_ds_points, num_samples=src_num_seeds, min_distance=radius)
        tgt_indices = furthest_point_sample(tgt_ds_points, num_samples=tgt_num_seeds, min_distance=radius)
        src_seeds = src_ds_points[src_indices]
        tgt_seeds = tgt_ds_points[tgt_indices]
        src_num_seeds = src_seeds.shape[0]
        tgt_num_seeds = tgt_seeds.shape[0]
        total_seeds = src_num_seeds + tgt_num_seeds

        # 1. compute features on the original point clouds
        lin_1, pla_1, sph_1, ani_1, var_1, ent_1, suf_1, com_1 = compute_point_features(
            src_seeds, tgt_seeds, src_points, tgt_points, radius, max_neighbors, voxel_size
        )

        # 2. compute features on the voxelized point clouds
        lin_2, pla_2, sph_2, ani_2, var_2, ent_2, suf_2, com_2 = compute_point_features(
            src_seeds, tgt_seeds, src_ds_points, tgt_ds_points, radius, max_neighbors, voxel_size
        )

        # 3. select the half seeds with the largest sphericity
        sel_indices = np.argsort(-sph_1)[: total_seeds // 2]

        # 3. absolute error
        abs_err_lin = np.abs(lin_1 - lin_2)
        abs_err_pla = np.abs(pla_1 - pla_2)
        abs_err_lin = abs_err_lin[sel_indices]
        abs_err_pla = abs_err_pla[sel_indices]
        abs_err = np.median(np.maximum(abs_err_lin, abs_err_pla))
        # abs_err_sph = np.abs(sph_1 - sph_2)
        # abs_err = np.median(np.maximum(abs_err_lin, abs_err_pla, abs_err_sph))

        # relative error
        rel_err_lin = np.abs(lin_1 - lin_2) / (np.maximum(np.abs(lin_1), np.abs(lin_2)) + 1e-8)
        rel_err_pla = np.abs(pla_1 - pla_2) / (np.maximum(np.abs(pla_1), np.abs(pla_2)) + 1e-8)
        rel_err_lin = rel_err_lin[sel_indices]
        rel_err_pla = rel_err_pla[sel_indices]
        rel_err = np.median(np.maximum(rel_err_lin, rel_err_pla))
        # rel_err_sph = np.abs(sph_1 - sph_2) / (np.maximum(np.abs(sph_1), np.abs(sph_2)) + 1e-8)
        # rel_err = np.median(np.maximum(rel_err_lin, rel_err_pla, rel_err_sph))

        # direction error
        dot_prod = np.sum(com_1 * com_2, axis=-1).clip(-1.0, 1.0)
        dot_prod = dot_prod[sel_indices]
        dir_err = np.median(np.arccos(dot_prod) / np.pi)

        # if abs_err < abs_tol and rel_err < rel_tol and dir_err < dir_tol:
        #     break

        if rel_err > rel_tol:
            break

        best_voxel_size = voxel_size
        voxel_size *= 1.0 + alpha

    best_voxel_size = max(best_voxel_size, min_voxel_size)
    best_voxel_size = min(best_voxel_size, max_voxel_size)

    return best_voxel_size


def geometry_aware_normalization(
    src_points: ndarray, tgt_points: ndarray, min_voxel_size: float | None = 0.02, max_voxel_size: float | None = None
) -> float:
    # 1. compute diagonal
    src_diagonal = get_bounding_box_diagonal(np.asarray(src_points))
    tgt_diagonal = get_bounding_box_diagonal(np.asarray(tgt_points))
    diagonal = min(src_diagonal, tgt_diagonal)

    # 2. compute initial voxel size
    # v1: pure linear projection
    # voxel_size = 0.23 / 199.0 * (diagonal - 1.0) + 0.02
    # voxel_size = 0.28 / 199.0 * (diagonal - 1.0) + 0.02

    # v2: linear + linear + log
    if diagonal >= 50.0:
        # keypoint: (50, 0.1), (200, 0.3)
        voxel_size = 0.1 * np.log2(diagonal / 25.0)
        # keypoint: (50, 0.1), (200, 0.25)
        # voxel_size = 0.075 * np.log2(diagonal / 50.0) + 0.1
    elif diagonal >= 25.0:
        # keypoint: (25, 0.075), (50, 0.1)
        voxel_size = 0.001 * (diagonal - 25.0) + 0.075
        # keypoint: (25, 0.05), (50, 0.1)
        # voxel_size = 0.002 * (diagonal - 25.0) + 0.05  # previous best
        # keypoint: (25, 0.045), (50, 0.1)
        # voxel_size = 0.0022 * (diagonal - 25.0) + 0.045
    else:
        # keypoint: (1, 0.015), (5, 0.025)
        voxel_size = 0.0025 * (diagonal - 1.0) + 0.015
        # keypoint: (1, 0.015), (5, 0.02)
        # voxel_size = 0.00125 * (diagonal - 1.0) + 0.015
        # keypoint: (2, 0.015), (5, 0.02)
        # voxel_size = 0.005 * (diagonal - 2.0) / 3.0 + 0.015
        # keypoint: (2.5, 0.015), (5, 0.02)
        # voxel_size = 0.002 * (diagonal - 2.5) + 0.015
        # keypoint: (0, 0.015), (5, 0.02)
        # voxel_size = 0.001 * diagonal + 0.015
        # keypoint: (0, 0.0175), (5, 0.02)
        # voxel_size = 0.0005 * diagonal + 0.0175

    # v3: linear + log + log
    # if diagonal >= 50.0:
    #     voxel_size = 0.05 * np.log2((diagonal / 25.0) ** 2)
    # elif diagonal >= 25.0:
    #     voxel_size = 0.05 * np.log2(diagonal / 12.5)
    # else:
    #     voxel_size = 0.05 / np.sqrt(-np.log2(diagonal / 50.0))

    # 3. adaptively update voxel size
    voxel_size = update_voxel_size(src_points, tgt_points, voxel_size)

    if min_voxel_size is not None:
        voxel_size = max(voxel_size, min_voxel_size)
    if max_voxel_size is not None:
        voxel_size = min(voxel_size, max_voxel_size)

    # print(f"diagonal: {diagonal:.3f}, voxel_size: {voxel_size:.6f}")

    return voxel_size
