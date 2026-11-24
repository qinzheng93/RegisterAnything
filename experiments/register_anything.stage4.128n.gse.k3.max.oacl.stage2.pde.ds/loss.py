import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from vision3d.loss import CircleLoss
from vision3d.ops.metrics import compute_isotropic_transform_error
from vision3d.ops.pairwise_distance import pairwise_distance
from vision3d.ops.se3 import apply_transform, scale_transform


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = CircleLoss(
            cfg.loss.coarse_loss.positive_margin,
            cfg.loss.coarse_loss.negative_margin,
            cfg.loss.coarse_loss.positive_optimal,
            cfg.loss.coarse_loss.negative_optimal,
            cfg.loss.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.loss.coarse_loss.positive_overlap

    def forward(self, output_dict: dict) -> Tensor:
        src_feats = output_dict["src_feats_c"]
        tgt_feats = output_dict["tgt_feats_c"]
        gt_src_node_corr_indices = output_dict["gt_src_node_corr_indices"]
        gt_tgt_node_corr_indices = output_dict["gt_tgt_node_corr_indices"]
        gt_node_corr_overlaps = output_dict["gt_node_corr_overlaps"]

        feat_dists = torch.sqrt(pairwise_distance(src_feats, tgt_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists, dtype=gt_node_corr_overlaps.dtype)
        overlaps[gt_src_node_corr_indices, gt_tgt_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.loss.fine_loss.positive_radius
        self.eps = 1e-9

    def forward(self, data_dict: dict, output_dict: dict) -> Tensor:
        src_node_corr_knn_points = output_dict["src_node_corr_knn_points"]  # (P, N, 3)
        tgt_node_corr_knn_points = output_dict["tgt_node_corr_knn_points"]  # (P, M, 3)
        src_node_corr_knn_scores = output_dict["src_node_corr_knn_scores"]  # (P, N)
        tgt_node_corr_knn_scores = output_dict["tgt_node_corr_knn_scores"]  # (P, M)
        src_node_corr_knn_masks = output_dict["src_node_corr_knn_masks"]  # (P, N)
        tgt_node_corr_knn_masks = output_dict["tgt_node_corr_knn_masks"]  # (P, M)
        matching_scores = output_dict["matching_scores"]  # (P, N, M)
        transform = data_dict["transform"]
        dataset_name = data_dict["dataset"]
        scaling_factor = data_dict["scale"]

        positive_radius = self.positive_radius[dataset_name] * scaling_factor

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dist_mat = pairwise_distance(src_node_corr_knn_points, tgt_node_corr_knn_points)  # (P, N, M)
        mask_mat = torch.logical_and(src_node_corr_knn_masks.unsqueeze(2), tgt_node_corr_knn_masks.unsqueeze(1))
        gt_corr_mat = torch.lt(dist_mat, positive_radius**2)
        gt_corr_mat = torch.logical_and(gt_corr_mat, mask_mat).float()

        # matchability loss
        score_loss = -(torch.log(matching_scores + self.eps) * gt_corr_mat).sum() / (gt_corr_mat.sum() + self.eps)

        # src & tgt confidence loss
        src_knn_labels = torch.logical_and(torch.eq(gt_corr_mat.sum(2), 0), src_node_corr_knn_masks).float()  # (P, N)
        tgt_knn_labels = torch.logical_and(torch.eq(gt_corr_mat.sum(1), 0), tgt_node_corr_knn_masks).float()  # (P, M)
        src_node_corr_knn_scores = 1.0 - src_node_corr_knn_scores + self.eps
        tgt_node_corr_knn_scores = 1.0 - tgt_node_corr_knn_scores + self.eps
        src_loss = -(torch.log(src_node_corr_knn_scores) * src_knn_labels).sum() / (src_knn_labels.sum() + self.eps)
        tgt_loss = -(torch.log(tgt_node_corr_knn_scores) * tgt_knn_labels).sum() / (tgt_knn_labels.sum() + self.eps)

        loss = score_loss + 0.5 * src_loss + 0.5 * tgt_loss

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.coarse_loss.weight
        self.weight_fine_loss = cfg.loss.fine_loss.weight

    def forward(self, data_dict: dict, output_dict: dict) -> dict:
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(data_dict, output_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            "loss": loss,
            "c_loss": coarse_loss,
            "f_loss": fine_loss,
        }


class EvalFunction(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.rmse_threshold = cfg.eval.rmse_threshold
        self.rre_threshold = cfg.eval.rre_threshold
        self.rte_threshold = cfg.eval.rte_threshold
        self.eps = 1e-9

    @torch.no_grad()
    def evaluate_coarse_matching(self, data_dict: dict, output_dict: dict) -> Tensor:
        src_length_c = output_dict["src_points_c"].shape[0]
        tgt_length_c = output_dict["tgt_points_c"].shape[0]
        gt_node_corr_overlaps = output_dict["gt_node_corr_overlaps"]
        gt_src_node_corr_indices = output_dict["gt_src_node_corr_indices"]
        gt_tgt_node_corr_indices = output_dict["gt_tgt_node_corr_indices"]
        src_node_corr_indices = output_dict["src_node_corr_indices"]
        tgt_node_corr_indices = output_dict["tgt_node_corr_indices"]

        acceptance_overlap = self.acceptance_overlap
        masks = torch.gt(gt_node_corr_overlaps, acceptance_overlap)
        gt_src_node_corr_indices = gt_src_node_corr_indices[masks]
        gt_tgt_node_corr_indices = gt_tgt_node_corr_indices[masks]
        gt_node_corr_map = torch.zeros(src_length_c, tgt_length_c).cuda()
        gt_node_corr_map[gt_src_node_corr_indices, gt_tgt_node_corr_indices] = 1.0

        precision = gt_node_corr_map[src_node_corr_indices, tgt_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine_matching(self, data_dict: dict, output_dict: dict) -> Tensor:
        dataset_name = data_dict["dataset"]
        scaling_factor = 1.0 / data_dict["scale"]
        transform = data_dict["transform"]
        src_corr_points = output_dict["src_corr_points"]
        tgt_corr_points = output_dict["tgt_corr_points"]

        # unscale transform and points
        transform = scale_transform(transform, scaling_factor)
        src_corr_points = src_corr_points * scaling_factor
        tgt_corr_points = tgt_corr_points * scaling_factor

        # compute correspondence distances
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(tgt_corr_points - src_corr_points, dim=1)

        # compute inlier ratio
        acceptance_radius = self.acceptance_radius[dataset_name]
        precision = torch.lt(corr_distances, acceptance_radius).float().mean()

        return precision

    @torch.no_grad()
    def evaluate_matchability_scores(self, data_dict: dict, output_dict: dict) -> tuple[Tensor, Tensor]:
        dataset_name = data_dict["dataset"]
        scaling_factor = 1.0 / data_dict["scale"]
        transform = data_dict["transform"]

        acceptance_radius = self.acceptance_radius[dataset_name]

        src_node_corr_knn_points = output_dict["src_node_corr_knn_points"]  # (P, N, 3)
        tgt_node_corr_knn_points = output_dict["tgt_node_corr_knn_points"]  # (P, M, 3)
        src_node_corr_knn_scores = output_dict["src_node_corr_knn_scores"]  # (P, N)
        tgt_node_corr_knn_scores = output_dict["tgt_node_corr_knn_scores"]  # (P, M)
        src_node_corr_knn_masks = output_dict["src_node_corr_knn_masks"]  # (P, N)
        tgt_node_corr_knn_masks = output_dict["tgt_node_corr_knn_masks"]  # (P, M)

        # unscale transform and points
        transform = scale_transform(transform, scaling_factor)
        src_node_corr_knn_points = src_node_corr_knn_points * scaling_factor
        tgt_node_corr_knn_points = tgt_node_corr_knn_points * scaling_factor

        # compute src & tgt knn labels
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dist_mat = pairwise_distance(src_node_corr_knn_points, tgt_node_corr_knn_points)  # (P, N, M)
        mask_mat = torch.logical_and(src_node_corr_knn_masks.unsqueeze(2), tgt_node_corr_knn_masks.unsqueeze(1))
        gt_corr_mat = torch.lt(dist_mat, acceptance_radius**2)
        gt_corr_mat = torch.logical_and(gt_corr_mat, mask_mat).float()
        src_knn_labels = torch.logical_and(torch.gt(gt_corr_mat.sum(2), 0), src_node_corr_knn_masks).float()  # (P, N)
        tgt_knn_labels = torch.logical_and(torch.gt(gt_corr_mat.sum(1), 0), tgt_node_corr_knn_masks).float()  # (P, M)

        # compute src & tgt knn predictions
        src_knn_results = torch.logical_and(torch.gt(src_node_corr_knn_scores, 0.5), src_node_corr_knn_masks).float()
        tgt_knn_results = torch.logical_and(torch.gt(tgt_node_corr_knn_scores, 0.5), tgt_node_corr_knn_masks).float()

        # true positives
        src_knn_n_true_positives = (src_knn_results * src_knn_labels).sum()
        tgt_knn_n_true_positives = (tgt_knn_results * tgt_knn_labels).sum()

        # precision & recall
        src_precision = src_knn_n_true_positives / (src_knn_results.sum() + self.eps)
        tgt_precision = tgt_knn_n_true_positives / (tgt_knn_results.sum() + self.eps)
        src_recall = src_knn_n_true_positives / (src_knn_labels.sum() + self.eps)
        tgt_recall = tgt_knn_n_true_positives / (tgt_knn_labels.sum() + self.eps)

        precision = 0.5 * (src_precision + tgt_precision)
        recall = 0.5 * (src_recall + tgt_recall)

        return precision, recall

    @torch.no_grad()
    def evaluate_registration(self, data_dict: dict, output_dict: dict) -> dict:
        dataset = data_dict["dataset"]
        scaling_factor = 1.0 / data_dict["scale"]
        transform = data_dict["transform"]
        est_transform = output_dict["estimated_transform"]
        src_points = output_dict["src_points"]

        # unscale transform and points
        transform = scale_transform(transform, scaling_factor)
        est_transform = scale_transform(est_transform, scaling_factor)
        src_points = src_points * scaling_factor

        result_dict = {}

        # compute rre and rte
        rre, rte = compute_isotropic_transform_error(transform, est_transform)

        result_dict["RRE"] = rre
        result_dict["RTE"] = rte

        if dataset == "3DMatch":
            # compute rmse
            realignment_transform = torch.matmul(torch.linalg.inv(transform), est_transform)
            realigned_src_points = apply_transform(src_points, realignment_transform)
            rmse = torch.linalg.norm(realigned_src_points - src_points, dim=1).mean()

            # compute recall
            rmse_threshold = self.rmse_threshold[dataset]
            recall = torch.lt(rmse, rmse_threshold).float()

            result_dict["RMSE"] = rmse
            result_dict["RR"] = recall
        else:
            # compute recall
            rre_threshold = self.rre_threshold[dataset]
            rte_threshold = self.rte_threshold[dataset]
            recall = ((rre < rre_threshold) & (rte < rte_threshold)).float()

            result_dict["RR"] = recall

        return result_dict

    def forward(self, data_dict: dict, output_dict: dict) -> dict:
        c_precision = self.evaluate_coarse_matching(data_dict, output_dict)
        f_precision = self.evaluate_fine_matching(data_dict, output_dict)
        m_precision, m_recall = self.evaluate_matchability_scores(data_dict, output_dict)
        reg_dict = self.evaluate_registration(data_dict, output_dict)

        result_dict = {}
        result_dict["PIR"] = c_precision
        result_dict["IR"] = f_precision
        result_dict["MPrec"] = m_precision
        result_dict["MRec"] = m_recall
        result_dict.update(reg_dict)

        return result_dict
