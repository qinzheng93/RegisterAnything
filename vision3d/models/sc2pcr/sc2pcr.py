import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops.eigenvector import leading_eigenvector
from vision3d.ops.pairwise_distance import pairwise_distance
from vision3d.ops.se3 import apply_transform
from vision3d.ops.spatial_consistency import cross_second_order_spatial_consistency, second_order_spatial_consistency
from vision3d.ops.weighted_procrustes import weighted_procrustes


class SC2PCR(nn.Module):
    """SC^2-PCR model.

    Paper: https://arxiv.org/abs/2203.14453.

    Ported from https://github.com/ZhiChen902/SC2-PCR.
    """

    def __init__(
        self,
        distance_threshold: float,
        max_correspondences: int = 8000,
        k1: int = 30,
        k2: int = 20,
        nms_radius: float = 0.1,
        seed_ratio: float = 0.2,
        tight_metric: bool = False,
        inlier_threshold: float = 0.1,
        max_iterations: int = 10,
        eps: float = 1e-6,
    ):
        """Initialize SC^2-PCR model.

        Args:
            sigma (float): The distance threshold parameter.
            max_correspondences (int, optional): The maximum number of correspondences. Defaults to 8000.
            k1 (int, optional): The number of neighbors for the first stage. Defaults to 30.
            k2 (int, optional): The number of neighbors for the second stage. Defaults to 20.
            nms_radius (float, optional): The radius for Non-Maximum Suppression (NMS). Defaults to 0.1.
            seed_ratio (float, optional): The ratio of seed correspondences. Defaults to 0.2.
            tight_metric (bool, optional): Whether to use tight metric. Defaults to False.
            inlier_threshold (float, optional): The threshold for inliers. Defaults to 0.1.
            max_iterations (int, optional): The maximum number of iterations for ICP. Defaults to 10.
            eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.distance_threshold = distance_threshold
        self.max_correspondences = max_correspondences
        self.k1 = k1
        self.k2 = k2
        self.nms_radius = nms_radius
        self.seed_ratio = seed_ratio
        self.tight_metric = tight_metric
        self.inlier_threshold = inlier_threshold
        self.max_iterations = max_iterations
        self.eps = eps

    def pick_seeds(self, dist_mat: Tensor, confidences: Tensor) -> Tensor:
        """Pick seed correspondences based on Parallel NMS.

        Args:
            dist_mat (Tensor): The distance matrix between correspondences in the source point clodu with shape (N, N).
            confidences (Tensor): The confidence scores for each correspondence with shape (N,).

        Returns:
            Tensor: The indices of seed correspondences with shape (K,).
        """
        num_corrs = dist_mat.shape[0]
        num_seeds = int(num_corrs * self.seed_ratio)

        nms_mat = torch.ge(confidences.unsqueeze(1), confidences.unsqueeze(0))  # (N, N)
        nms_mat = torch.logical_or(nms_mat, torch.ge(dist_mat, self.nms_radius))  # (N, N)
        # nms_mat_{i,j} is False means that there exists corr_j which has higher confidence than corr_i and their distance is less than the NMS radius, so corr_i should be removed.
        is_local_max = nms_mat.float().min(dim=-1)[0]

        confidences_max = confidences * is_local_max
        _, sel_indices = confidences_max.topk(num_seeds, dim=0, largest=True, sorted=True)

        return sel_indices

    def compute_seed_transforms(
        self, src_corr_points: Tensor, tgt_corr_points: Tensor, seed_indices: Tensor, seed_sc2_mat: Tensor
    ) -> Tensor:
        """Compute transformations for all seeds.

        Args:
            src_corr_points (Tensor): The correspondences in the source point cloud with shape (N, 3).
            tgt_corr_points (Tensor): The correspondences in the target point cloud with shape (N, 3).
            seed_indices (Tensor): The indices of seed correspondences with shape (K,).
            seed_sc2_mat (Tensor): The 2-order SC matrix for seed correspondences with shape (K, N).

        Returns:
            Tensor: The estimated transform with shape (4, 4).
        """
        num_corrs = src_corr_points.shape[0]

        # limit the number of neighbors if there are too few correspondences.
        k1 = self.k1 if self.k1 <= num_corrs else 4
        k2 = self.k2 if self.k1 <= num_corrs else 4

        # seed correspondences
        seed_src_corr_points = src_corr_points[seed_indices]  # (K, 3)
        seed_tgt_corr_points = tgt_corr_points[seed_indices]  # (K, 3)

        # first stage: select k1 neighbors
        _, s1_local_indices = seed_sc2_mat.topk(k1, dim=-1, largest=True, sorted=True)  # (K, k1)
        s1_local_src_corr_points = src_corr_points[s1_local_indices]  # (K, k1, 3)
        s1_local_tgt_corr_points = tgt_corr_points[s1_local_indices]  # (K, k1, 3)
        # NOTE: our implementation is slightly different from the original code:
        # 1. The original code does not multiply the outer elementwise product, but we apply it.
        # 2. The original code computes SC2 with k1 correspondences, and then select the first one as seed-to-neighbor
        #    metric, this could be problematic as it is likely that the first one is not the seed. We directly compute
        #    SC2 between the seed and the local correpondences.
        s1_local_sc2_mat = cross_second_order_spatial_consistency(
            seed_src_corr_points.unsqueeze(1),
            seed_tgt_corr_points.unsqueeze(1),
            s1_local_src_corr_points,
            s1_local_tgt_corr_points,
            self.distance_threshold,
        )  # (K, 1, k1)
        s1_local_sc2_mat = s1_local_sc2_mat.squeeze(1)  # (K, k1)

        # second stage: select k2 neighbors out of k1 neighbors
        _, s2_local_indices = s1_local_sc2_mat.topk(k2, dim=-1, largest=True, sorted=True)  # (K, k2)
        s2_local_indices = torch.gather(s1_local_indices, dim=1, index=s2_local_indices)  # (K, k2)
        s2_local_src_corr_points = src_corr_points[s2_local_indices]  # (K, k2, 3)
        s2_local_tgt_corr_points = tgt_corr_points[s2_local_indices]  # (K, k2, 3)
        s2_local_sc2_mat = second_order_spatial_consistency(
            s2_local_src_corr_points, s2_local_tgt_corr_points, self.distance_threshold * 2
        )  # (K, k2, k2)
        # NOTE: our implementation is slightly different from the original code:
        # 1. We use normalized hard SC2 instead of SC1 as the metric. But we do not notice obvious performance differences.
        # s2_local_sc2_mat = s2_local_sc1_mat  # use SC1 metric
        s2_local_sc2_mat = s2_local_sc2_mat / k2  # use normalized SC2 metric
        # s2_local_sc2_mat = s2_local_soft_sc2_mat  # use soft SC2 metric

        # compute leading eigenvector
        digonal_masks = 1.0 - torch.eye(k2).unsqueeze(0).cuda()  # (K, k2, k2)
        s2_local_sc2_mat = s2_local_sc2_mat * digonal_masks  # (K, k2, k2)
        s2_local_weights = leading_eigenvector(s2_local_sc2_mat)  # (K, k2)

        # compute transformation
        s2_local_transforms = weighted_procrustes(
            s2_local_src_corr_points, s2_local_tgt_corr_points, weights=s2_local_weights
        )  # (K, 4, 4)

        # hypothesis selection
        tsfm_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), s2_local_transforms)  # (K, N, 3)
        deltas = torch.linalg.norm(tsfm_src_corr_points - tgt_corr_points.unsqueeze(0), dim=-1)  # (K, N)
        inlier_counts = torch.lt(deltas, self.inlier_threshold).long().sum(dim=-1)  # (K,)
        best_index = inlier_counts.argmax()  # (1,)
        best_transform = s2_local_transforms[best_index]  # (4, 4)

        return best_transform

    def post_refine(self, src_corr_points: Tensor, tgt_corr_points: Tensor, transform: Tensor) -> Tensor:
        """Post-refinement of the transform with ICP.

        Args:
            src_corr_points (Tensor): The correspondences in the source point cloud with shape (N, 3).
            tgt_corr_points (Tensor): The correspondences in the target point cloud with shape (N, 3).
            transform (Tensor): The initial transform with shape (4, 4).

        Returns:
            Tensor: The refined transform with shape (4, 4).
        """
        last_inlier_count = 0
        for _ in range(self.max_iterations):
            # transform source points
            tsfm_src_corr_points = apply_transform(src_corr_points, transform)  # (N, 3)

            # compute inlier count
            distances = torch.linalg.norm(tsfm_src_corr_points - tgt_corr_points, dim=-1)  # (N,)
            inlier_masks = torch.lt(distances, self.inlier_threshold)  # (N,)
            inlier_count = inlier_masks.long().sum().item()  # ()

            # early stop
            if inlier_count <= last_inlier_count:
                break

            if inlier_count < 3:
                break

            last_inlier_count = inlier_count

            # update transform
            cur_src_corr_points = src_corr_points[inlier_masks]
            cur_tgt_corr_points = tgt_corr_points[inlier_masks]
            cur_weights = 1.0 / (1.0 + distances[inlier_masks] ** 2 / self.inlier_threshold**2)

            # update transform
            transform = weighted_procrustes(cur_src_corr_points, cur_tgt_corr_points, weights=cur_weights)  # (4, 4)

        return transform

    def forward(self, src_corr_points: Tensor, tgt_corr_points: Tensor) -> Tensor:
        """SC^2-PCR forward propagation.

        Args:
            src_corr_points (Tensor): The correspondences in the source point cloud with shape (N, 3).
            tgt_corr_points (Tensor): The correspondences in the target point cloud with shape (N, 3).

        Returns:
            Tensor: The estimated transform with shape (4, 4).
        """
        if src_corr_points.shape[0] > self.max_correspondences:
            src_corr_points = src_corr_points[: self.max_correspondences]
            tgt_corr_points = tgt_corr_points[: self.max_correspondences]

        # pairwise distance
        src_dist_mat = pairwise_distance(src_corr_points, src_corr_points, squared=False)  # (N, N)
        tgt_dist_mat = pairwise_distance(tgt_corr_points, tgt_corr_points, squared=False)  # (N, N)
        delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)  # (N, N)

        # 1-order SC
        sc1_mat = torch.relu(1.0 - delta_mat**2 / self.distance_threshold**2)
        hard_sc1_mat = torch.lt(delta_mat, self.distance_threshold).float()  # (N, N)

        # compute confidence
        confidences = leading_eigenvector(sc1_mat)
        seed_indices = self.pick_seeds(src_dist_mat, confidences)  # (K,)

        # 2-order SC
        seed_hard_sc1_mat = hard_sc1_mat[seed_indices]  # (K, N)
        if self.tight_metric:
            hard_sc1_mat_tight = torch.lt(delta_mat, 0.5 * self.distance_threshold).float()  # (N, N)
            seed_hard_sc1_mat_tight = hard_sc1_mat_tight[seed_indices]  # (K, N)
            seed_sc2_mat = seed_hard_sc1_mat * torch.matmul(seed_hard_sc1_mat_tight, hard_sc1_mat_tight)  # (K, N)
        else:
            seed_sc2_mat = seed_hard_sc1_mat * torch.matmul(seed_hard_sc1_mat, hard_sc1_mat)  # (K, N)

        # compute transformations for all seeds
        transform = self.compute_seed_transforms(src_corr_points, tgt_corr_points, seed_indices, seed_sc2_mat)

        # post refinement
        transform = self.post_refine(src_corr_points, tgt_corr_points, transform)

        return transform
