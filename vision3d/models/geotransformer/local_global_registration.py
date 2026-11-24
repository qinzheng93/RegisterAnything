import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers.weighted_procrustes import WeightedProcrustes
from vision3d.ops.mutual_topk_select import batch_mutual_topk_select
from vision3d.ops.pairwise_distance import pairwise_distance
from vision3d.ops.se3 import apply_transform


def convert_to_batch(
    src_corr_points: Tensor, tgt_corr_points: Tensor, corr_scores: Tensor, chunks: list[tuple[int, int]]
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert stacked correspondences to batched points.

    The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
    transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
    into a batch.

    Args:
        src_corr_points (Tensor): (C, 3)
        tgt_corr_points (Tensor): (C, 3)
        corr_scores (Tensor): (C,)
        chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

    Returns:
        batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
        batch_tgt_corr_points (Tensor): (B, K, 3), padded with zeros.
        batch_corr_scores (Tensor): (B, K), padded with zeros.
    """
    batch_size = len(chunks)
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    src_corr_points = src_corr_points[indices]  # (total, 3)
    tgt_corr_points = tgt_corr_points[indices]  # (total, 3)
    corr_scores = corr_scores[indices]  # (total,)

    max_corr = np.max([y - x for x, y in chunks])
    target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
    indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
    indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
    indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

    batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
    batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
    batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

    batch_tgt_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
    batch_tgt_corr_points.index_put_([indices0, indices1], tgt_corr_points)
    batch_tgt_corr_points = batch_tgt_corr_points.view(batch_size, max_corr, 3)

    batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
    batch_corr_scores.index_put_([indices], corr_scores)
    batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

    return batch_src_corr_points, batch_tgt_corr_points, batch_corr_scores


def recompute_correspondence_scores(
    src_corr_points: Tensor, tgt_corr_points: Tensor, estimated_transform: Tensor, acceptance_radius: float = 1.0
) -> Tensor:
    aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
    corr_residuals = torch.linalg.norm(tgt_corr_points - aligned_src_corr_points, dim=1)
    inlier_masks = torch.lt(corr_residuals, acceptance_radius)
    corr_scores = 1.0 / (1.0 + corr_residuals**2 / acceptance_radius**2)
    corr_scores = corr_scores * inlier_masks.float()
    return corr_scores


class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float | None = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        use_logits: bool = True,
        use_nms: bool = False,
        nms_radius: float | None = None,
        min_local_correspondences: int = 3,
        max_global_correspondences: int | None = None,
        filter_correspondences: bool = False,
        num_refinement_steps: int = 5,
        decay_factor: float = 0.9,
    ):
        """Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool): mutual or non-mutual matching. Default: True.
            confidence_threshold (float): ignore matches whose scores are below this threshold. Default: 0.05.
            use_dustbin (bool): if True, the dustbin row/column is used in the score matrix. Default: False.
            use_global_score (bool): if True, use patch correspondence scores. Default: False.
            use_logits (bool): if True, the score matrix is logits (requires exp). Default: True.
            use_nms (bool): if True, use FastNMS to remove close correspondences. Default: False.
            nms_radius (float, optional): the radius of nms. Default: None.
            min_local_correspondences (int): the minimal correspondences for each patch correspondence. Default: 3.
            max_global_correspondences (int, optional): the maximal global correspondences. Default: None.
            num_refinement_steps (int): the number of refinement steps. Default: 5.
            decay_factor (float): the decay factor of acceptance radius for each refinement step. Default: 0.9.
        """
        super().__init__()

        assert not use_nms or nms_radius is not None, "nms_radius must be specified when use_nms is True"

        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.use_logits = use_logits
        self.use_nms = use_nms
        self.nms_radius = nms_radius
        self.min_local_correspondences = min_local_correspondences
        self.max_global_correspondences = max_global_correspondences
        self.num_refinement_steps = num_refinement_steps
        self.decay_factor = decay_factor
        self.filter_correspondences = filter_correspondences

        self.procrustes = WeightedProcrustes()

    def local_to_global_registration(
        self, src_knn_points: Tensor, tgt_knn_points: Tensor, score_mat: Tensor, corr_mat: Tensor
    ) -> tuple[Tensor, ...]:
        # extract point correspondences
        batch_indices, src_indices, tgt_indices = torch.nonzero(corr_mat, as_tuple=True)
        all_src_corr_points = src_knn_points[batch_indices, src_indices]
        all_tgt_corr_points = tgt_knn_points[batch_indices, tgt_indices]
        all_corr_scores = score_mat[batch_indices, src_indices, tgt_indices]

        # build verification set
        global_corr_scores = all_corr_scores
        global_src_corr_points = all_src_corr_points
        global_tgt_corr_points = all_tgt_corr_points

        # remove close correspondences with NMS
        if self.use_nms and self.nms_radius is not None:
            src_dist_mat = pairwise_distance(all_src_corr_points, all_src_corr_points)
            tgt_dist_mat = pairwise_distance(all_tgt_corr_points, all_tgt_corr_points)
            src_dist_mask_mat = torch.lt(src_dist_mat, self.nms_radius**2)
            tgt_dist_mask_mat = torch.lt(tgt_dist_mat, self.nms_radius**2)
            dist_mask_mat = torch.logical_and(src_dist_mask_mat, tgt_dist_mask_mat)
            score_mask_mat = torch.lt(all_corr_scores.unsqueeze(1), all_corr_scores.unsqueeze(0))
            masks = torch.eq(torch.logical_and(dist_mask_mat, score_mask_mat).float().sum(1), 0)
            global_corr_scores = all_corr_scores[masks]
            global_src_corr_points = all_src_corr_points[masks]
            global_tgt_corr_points = all_tgt_corr_points[masks]

        # select top-k correspondences if there are too many correspondences
        num_corrs = global_corr_scores.shape[0]
        if self.max_global_correspondences is not None and num_corrs > self.max_global_correspondences:
            global_corr_scores, sel_indices = global_corr_scores.topk(k=self.max_global_correspondences, largest=True)
            global_src_corr_points = global_src_corr_points[sel_indices]
            global_tgt_corr_points = global_tgt_corr_points[sel_indices]

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        all_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        all_indices = torch.nonzero(all_masks, as_tuple=True)[0] + 1
        all_indices = all_indices.detach().cpu().numpy().tolist()
        all_indices = [0] + all_indices + [batch_indices.shape[0]]
        chunks = [(x, y) for x, y in zip(all_indices[:-1], all_indices[1:]) if y - x >= self.min_local_correspondences]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            local_src_corr_points, local_tgt_corr_points, local_corr_scores = convert_to_batch(
                all_src_corr_points, all_tgt_corr_points, all_corr_scores, chunks
            )
            local_transforms = self.procrustes(local_src_corr_points, local_tgt_corr_points, local_corr_scores)
            local_aligned_src_corr_points = apply_transform(global_src_corr_points.unsqueeze(0), local_transforms)
            local_corr_residuals = torch.linalg.norm(
                global_tgt_corr_points.unsqueeze(0) - local_aligned_src_corr_points, dim=2
            )
            local_inlier_masks = torch.lt(local_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = local_inlier_masks.sum(dim=1).argmax()
            estimated_transform = local_transforms[best_index]
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, global_corr_scores)

        # filter out outliers based on the initial transformation
        if self.filter_correspondences:
            global_corr_scores = recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, estimated_transform, 1.5 * self.acceptance_radius
            )
            inlier_masks = torch.gt(global_corr_scores, 0.0)
            if inlier_masks.sum() > 0:
                global_src_corr_points = global_src_corr_points[inlier_masks]
                global_tgt_corr_points = global_tgt_corr_points[inlier_masks]
                global_corr_scores = global_corr_scores[inlier_masks]

        # global refinement
        acceptance_radius = self.acceptance_radius
        for _ in range(self.num_refinement_steps):
            global_corr_scores = recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, estimated_transform, acceptance_radius
            )
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, global_corr_scores)
            acceptance_radius *= self.decay_factor

        return global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        src_knn_points: Tensor,
        tgt_knn_points: Tensor,
        src_knn_masks: Tensor,
        tgt_knn_masks: Tensor,
        score_mat: Tensor,
        global_scores: Tensor,
    ) -> tuple[Tensor, ...]:
        """Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            src_knn_points (Tensor): The points in the source patches (B, K, 3).
            tgt_knn_points (Tensor): The points in the target patches (B, K, 3).
            src_knn_masks (Tensor[bool]): The point masks in the source patches, True if valid (B, K).
            tgt_knn_masks (Tensor[bool]): The point masks in the target patches, True if valid (B, K).
            score_mat (Tensor): The log likelihood matrix for each patch pair (B, K, K) or (B, K + 1, K + 1).
            global_scores (Tensor): The matching scores for each patch pair (B,).

        Returns:
            src_corr_indices (LongTensor): The indices of the correspondence points in the source point cloud (C,).
            tgt_corr_indices (LongTensor): The indices of the correspondence points in the target point cloud (C,).
            src_corr_points (Tensor): The correspondence points in the source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in the target point cloud (C, 3).
            corr_scores (Tensor): The matching scores of the correspondences (C,).
            estimated_transform (Tensor): The estimated transformation from source to target (4, 4).
        """
        if self.use_logits:
            score_mat = torch.exp(score_mat)

        corr_mat = batch_mutual_topk_select(
            score_mat,
            self.k,
            row_masks=src_knn_masks,
            col_masks=tgt_knn_masks,
            threshold=self.confidence_threshold,
            mutual=self.mutual,
            reduce_result=False,
        )  # (B, K, K)

        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        src_corr_points, tgt_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            src_knn_points, tgt_knn_points, score_mat, corr_mat
        )

        return src_corr_points, tgt_corr_points, corr_scores, estimated_transform

    def extra_repr(self) -> str:
        param_strings = []
        param_strings.append(f"k={self.k}")
        param_strings.append(f"radius={self.acceptance_radius:g}")
        param_strings.append(f"mutual={self.mutual}")
        if self.confidence_threshold is not None:
            param_strings.append(f"threshold={self.confidence_threshold:g}")
        else:
            param_strings.append(f"threshold={self.confidence_threshold}")
        param_strings.append(f"use_dustbin={self.use_dustbin}")
        param_strings.append(f"use_global_score={self.use_global_score}")
        param_strings.append(f"use_logits={self.use_logits}")
        param_strings.append(f"use_nms={self.use_nms}")
        if self.nms_radius is not None:
            param_strings.append(f"nms_radius={self.nms_radius:g}")
        else:
            param_strings.append(f"nms_radius={self.nms_radius}")
        param_strings.append(f"min_local_correspondences={self.min_local_correspondences}")
        param_strings.append(f"max_global_correspondences={self.max_global_correspondences}")
        param_strings.append(f"filter_correspondences={self.filter_correspondences}")
        param_strings.append(f"refinement={self.num_refinement_steps}")
        format_string = ", ".join(param_strings)
        return format_string


class LocalGlobalRegistrationWithDuplicateRemoval(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float | None = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        use_logits: bool = True,
        min_local_correspondences: int = 3,
        max_global_correspondences: int | None = None,
        num_refinement_steps: int = 5,
    ):
        """Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            min_local_correspondences (int=3): minimal number of correspondences for each patch correspondence.
            max_global_correspondences (int=None): maximal number of global correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super().__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.use_logits = use_logits
        self.min_local_correspondences = min_local_correspondences
        self.max_global_correspondences = max_global_correspondences
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes()

    def local_to_global_registration(
        self,
        src_points: Tensor,
        tgt_points: Tensor,
        src_knn_indices: Tensor,
        tgt_knn_indices: Tensor,
        score_mat: Tensor,
        corr_mat: Tensor,
    ) -> tuple[Tensor, ...]:
        # extract point correspondences
        batch_indices, src_indices, tgt_indices = torch.nonzero(corr_mat, as_tuple=True)

        all_src_corr_indices = src_knn_indices[batch_indices, src_indices]
        all_tgt_corr_indices = tgt_knn_indices[batch_indices, tgt_indices]
        all_src_corr_points = src_points[all_src_corr_indices]
        all_tgt_corr_points = tgt_points[all_tgt_corr_indices]
        all_corr_scores = score_mat[batch_indices, src_indices, tgt_indices]

        # duplicate removal
        all_corr_indices = torch.stack([all_src_corr_indices, all_tgt_corr_indices], dim=1)
        global_corr_indices, inv_indices, unique_counts = torch.unique(
            all_corr_indices, return_inverse=True, return_counts=True, dim=0
        )
        global_src_corr_indices = global_corr_indices[:, 0]
        global_tgt_corr_indices = global_corr_indices[:, 1]
        global_src_corr_points = src_points[global_src_corr_indices]
        global_tgt_corr_points = tgt_points[global_tgt_corr_indices]
        global_corr_scores = torch.zeros(size=(global_corr_indices.shape[0],)).cuda()
        global_corr_scores.scatter_reduce_(dim=0, index=inv_indices, src=all_corr_scores, reduce="amax")

        # build global correspondences
        if (
            self.max_global_correspondences is not None
            and global_corr_scores.shape[0] > self.max_global_correspondences
        ):
            global_corr_scores, sel_indices = global_corr_scores.topk(k=self.max_global_correspondences, largest=True)
            global_src_corr_points = global_src_corr_points[sel_indices]
            global_tgt_corr_points = global_tgt_corr_points[sel_indices]

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        all_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        all_indices = torch.nonzero(all_masks, as_tuple=True)[0] + 1
        all_indices = all_indices.detach().cpu().numpy().tolist()
        all_indices = [0] + all_indices + [batch_indices.shape[0]]
        chunks = [(x, y) for x, y in zip(all_indices[:-1], all_indices[1:]) if y - x >= self.min_local_correspondences]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            local_src_corr_points, local_tgt_corr_points, local_corr_scores = convert_to_batch(
                all_src_corr_points, all_tgt_corr_points, all_corr_scores, chunks
            )
            local_transforms = self.procrustes(local_src_corr_points, local_tgt_corr_points, local_corr_scores)
            local_aligned_src_corr_points = apply_transform(global_src_corr_points.unsqueeze(0), local_transforms)
            local_corr_residuals = torch.linalg.norm(
                global_tgt_corr_points.unsqueeze(0) - local_aligned_src_corr_points, dim=2
            )
            local_inlier_masks = torch.lt(local_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = local_inlier_masks.sum(dim=1).argmax()
            estimated_transform = local_transforms[best_index]
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, global_corr_scores)

        # global refinement
        for _ in range(self.num_refinement_steps):
            cur_corr_scores = recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, estimated_transform, self.acceptance_radius
            )
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, cur_corr_scores)

        return global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        src_points: Tensor,
        tgt_points: Tensor,
        src_knn_indices: Tensor,
        tgt_knn_indices: Tensor,
        src_knn_masks: Tensor,
        tgt_knn_masks: Tensor,
        score_mat: Tensor,
        global_scores: Tensor,
    ) -> tuple[Tensor, ...]:
        """Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            src_points (Tensor): The points in the source point cloud (N, 3).
            tgt_points (Tensor): The points in the target point cloud (M, 3).
            src_knn_indices (LongTensor): The indices of the points in the source patches (B, K).
            tgt_knn_indices (LongTensor): The indices of the points in the target patches (B, K).
            src_knn_masks (Tensor[bool]): The point masks in the source patches, True if valid (B, K).
            tgt_knn_masks (Tensor[bool]): The point masks in the target patches, True if valid (B, K).
            score_mat (Tensor): The log likelihood matrix for each patch pair (B, K, K) or (B, K + 1, K + 1).
            global_scores (Tensor): The matching scores for each patch pair (B,).

        Returns:
            src_corr_indices (LongTensor): The indices of the correspondence points in the source point cloud (C,).
            tgt_corr_indices (LongTensor): The indices of the correspondence points in the target point cloud (C,).
            src_corr_points (Tensor): The correspondence points in the source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in the target point cloud (C, 3).
            corr_scores (Tensor): The matching scores of the correspondences (C,).
            estimated_transform (Tensor): The estimated transformation from source to target (4, 4).
        """
        if self.use_logits:
            score_mat = torch.exp(score_mat)

        corr_mat = batch_mutual_topk_select(
            score_mat,
            self.k,
            row_masks=src_knn_masks,
            col_masks=tgt_knn_masks,
            threshold=self.confidence_threshold,
            mutual=self.mutual,
            reduce_result=False,
        )  # (B, K, K)

        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        src_corr_points, tgt_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            src_points, tgt_points, src_knn_indices, tgt_knn_indices, score_mat, corr_mat
        )

        return src_corr_points, tgt_corr_points, corr_scores, estimated_transform

    def extra_repr(self) -> str:
        param_strings = []
        param_strings.append(f"k={self.k}")
        param_strings.append(f"radius={self.acceptance_radius:g}")
        param_strings.append(f"mutual={self.mutual}")
        if self.confidence_threshold is not None:
            param_strings.append(f"threshold={self.confidence_threshold:g}")
        else:
            param_strings.append(f"threshold={self.confidence_threshold}")
        param_strings.append(f"use_dustbin={self.use_dustbin}")
        param_strings.append(f"use_global_score={self.use_global_score}")
        param_strings.append(f"use_logits={self.use_logits}")
        param_strings.append(f"use_nms={self.use_nms}")
        if self.nms_radius is not None:
            param_strings.append(f"nms_radius={self.nms_radius:g}")
        else:
            param_strings.append(f"nms_radius={self.nms_radius}")
        param_strings.append(f"min_local_correspondences={self.min_local_correspondences}")
        param_strings.append(f"max_global_correspondences={self.max_global_correspondences}")
        param_strings.append(f"filter_correspondences={self.filter_correspondences}")
        param_strings.append(f"refinement={self.num_refinement_steps}")
        format_string = ", ".join(param_strings)
        return format_string


class LocalGlobalRegistrationThreshold(nn.Module):
    def __init__(
        self,
        acceptance_radius: float,
        confidence_threshold: float = 0.2,
        min_local_correspondences: int = 3,
        min_global_correspondences: int = 256,
        max_global_correspondences: int | None = None,
        num_refinement_steps: int = 5,
    ):
        """Point Matching with Local-to-Global Registration.

        Args:
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            min_local_correspondences (int=3): minimal number of correspondences for each patch correspondence.
            max_global_correspondences (int=None): maximal number of global correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super().__init__()
        self.acceptance_radius = acceptance_radius
        self.confidence_threshold = confidence_threshold
        self.min_local_correspondences = min_local_correspondences
        self.min_global_correspondences = min_global_correspondences
        self.max_global_correspondences = max_global_correspondences
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes()

    def recompute_correspondence_scores(
        self, src_corr_points: Tensor, tgt_corr_points: Tensor, corr_scores: Tensor, estimated_transform: Tensor
    ) -> Tensor:
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(tgt_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def local_to_global_registration(
        self, src_knn_points: Tensor, tgt_knn_points: Tensor, score_mat: Tensor, corr_mat: Tensor
    ) -> tuple[Tensor, ...]:
        # extract point correspondences
        batch_indices, src_indices, tgt_indices = torch.nonzero(corr_mat, as_tuple=True)
        all_src_corr_points = src_knn_points[batch_indices, src_indices]
        all_tgt_corr_points = tgt_knn_points[batch_indices, tgt_indices]
        all_corr_scores = score_mat[batch_indices, src_indices, tgt_indices]

        # build verification set
        if self.max_global_correspondences is not None and all_corr_scores.shape[0] > self.max_global_correspondences:
            global_corr_scores, sel_indices = all_corr_scores.topk(k=self.max_global_correspondences, largest=True)
            global_src_corr_points = all_src_corr_points[sel_indices]
            global_tgt_corr_points = all_tgt_corr_points[sel_indices]
        else:
            global_src_corr_points = all_src_corr_points
            global_tgt_corr_points = all_tgt_corr_points
            global_corr_scores = all_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        all_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        all_indices = torch.nonzero(all_masks, as_tuple=True)[0] + 1
        all_indices = all_indices.detach().cpu().numpy().tolist()
        all_indices = [0] + all_indices + [batch_indices.shape[0]]
        chunks = [(x, y) for x, y in zip(all_indices[:-1], all_indices[1:]) if y - x >= self.min_local_correspondences]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            local_src_corr_points, local_tgt_corr_points, local_corr_scores = convert_to_batch(
                all_src_corr_points, all_tgt_corr_points, all_corr_scores, chunks
            )
            local_transforms = self.procrustes(local_src_corr_points, local_tgt_corr_points, local_corr_scores)
            local_aligned_src_corr_points = apply_transform(global_src_corr_points.unsqueeze(0), local_transforms)
            local_corr_residuals = torch.linalg.norm(
                global_tgt_corr_points.unsqueeze(0) - local_aligned_src_corr_points, dim=2
            )
            local_inlier_masks = torch.lt(local_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = local_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = global_corr_scores * local_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, global_corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform
            )

        # global refinement
        for _ in range(self.num_refinement_steps - 1):
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, cur_corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform
            )
        estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, cur_corr_scores)

        return global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        src_knn_points: Tensor,
        tgt_knn_points: Tensor,
        src_knn_masks: Tensor,
        tgt_knn_masks: Tensor,
        score_mat: Tensor,
    ) -> tuple[Tensor, ...]:
        """Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            src_knn_points (Tensor): The points in the source patches (B, K, 3).
            tgt_knn_points (Tensor): The points in the target patches (B, K, 3).
            src_knn_masks (Tensor[bool]): The point masks in the source patches, True if valid (B, K).
            tgt_knn_masks (Tensor[bool]): The point masks in the target patches, True if valid (B, K).
            score_mat (Tensor): The log likelihood matrix for each patch pair (B, K, K) or (B, K + 1, K + 1).

        Returns:
            src_corr_indices (LongTensor): The indices of the correspondence points in the source point cloud (C,).
            tgt_corr_indices (LongTensor): The indices of the correspondence points in the target point cloud (C,).
            src_corr_points (Tensor): The correspondence points in the source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in the target point cloud (C, 3).
            corr_scores (Tensor): The matching scores of the correspondences (C,).
            estimated_transform (Tensor): The estimated transformation from source to target (4, 4).
        """
        score_mat = torch.sigmoid(score_mat)
        mask_mat = torch.logical_and(src_knn_masks.unsqueeze(2), tgt_knn_masks.unsqueeze(1))

        confidence_threshold = self.confidence_threshold
        min_global_correspondences = min(score_mat.shape[1], score_mat.shape[2]) * 3
        min_global_correspondences = min(min_global_correspondences, self.min_global_correspondences)
        min_global_correspondences = min(min_global_correspondences, mask_mat.sum().item())
        while True:
            corr_mat = torch.gt(score_mat, confidence_threshold)
            corr_mat = torch.logical_and(corr_mat, mask_mat)
            if corr_mat.sum() >= min_global_correspondences:
                break
            confidence_threshold -= 0.05

        score_mat = score_mat * corr_mat.float()

        src_corr_points, tgt_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            src_knn_points, tgt_knn_points, score_mat, corr_mat
        )

        return src_corr_points, tgt_corr_points, corr_scores, estimated_transform

    def extra_repr(self) -> str:
        param_strings = [
            f"radius={self.acceptance_radius:g}",
            f"threshold={self.confidence_threshold:g}",
            f"min_local_corr={self.min_local_correspondences}",
            f"min_global_corr={self.min_global_correspondences}",
            f"max_global_corr={self.max_global_correspondences}",
            f"refinement={self.num_refinement_steps}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
