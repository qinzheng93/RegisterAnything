import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Module

from vision3d.models.geotransformer import (
    DualSoftmaxWithConfidence,
    GeometricTransformer,
    LocalGeometricTransformer,
    LocalGlobalRegistration,
    SuperPointMatchingGlobalTopk,
    SuperPointProposalGenerator,
)
from vision3d.ops.index_select import index_select
from vision3d.ops.matching import get_patch_correspondences
from vision3d.ops.point_cloud_partition import point_to_node_partition

# isort: split
from backbone import KPConvFPN


class GeoTransformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.matching_radius

        self.backbone = KPConvFPN(
            cfg.model.backbone.input_dim,
            cfg.model.backbone.output_dim,
            cfg.model.backbone.init_dim,
            cfg.model.backbone.kernel_size,
            cfg.model.backbone.base_radius,
            cfg.model.backbone.base_sigma,
        )

        self.c_transformer = LocalGeometricTransformer(
            cfg.model.c_geotransformer.input_dim,
            cfg.model.c_geotransformer.output_dim,
            cfg.model.c_geotransformer.hidden_dim,
            cfg.model.c_geotransformer.num_heads,
            cfg.model.c_geotransformer.num_neighbors,
            cfg.model.c_geotransformer.blocks,
            cfg.model.c_geotransformer.sigma_d,
            sigma_a=cfg.model.c_geotransformer.sigma_a,
            angle_k=cfg.model.c_geotransformer.angle_k,
            reduction_a=cfg.model.c_geotransformer.angle_reduction,
            embedding=cfg.model.c_geotransformer.embedding,
            norm_cfg=cfg.model.c_geotransformer.norm_cfg,
            act_cfg=cfg.model.c_geotransformer.act_cfg,
        )

        self.coarse_target = SuperPointProposalGenerator(
            cfg.model.coarse_matching.num_targets, cfg.model.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatchingGlobalTopk(
            cfg.model.coarse_matching.num_correspondences, cfg.model.coarse_matching.dual_normalization
        )

        self.f_transformer = GeometricTransformer(
            cfg.model.f_geotransformer.input_dim,
            cfg.model.f_geotransformer.output_dim,
            cfg.model.f_geotransformer.hidden_dim,
            cfg.model.f_geotransformer.num_heads,
            cfg.model.f_geotransformer.blocks,
            cfg.model.f_geotransformer.sigma_d,
            embedding=cfg.model.f_geotransformer.embedding,
            norm_cfg=cfg.model.f_geotransformer.norm_cfg,
            act_cfg=cfg.model.f_geotransformer.act_cfg,
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.model.fine_matching.topk,
            cfg.model.fine_matching.acceptance_radius,
            mutual=cfg.model.fine_matching.mutual,
            confidence_threshold=cfg.model.fine_matching.confidence_threshold,
            use_dustbin=cfg.model.fine_matching.use_dustbin,
            use_global_score=cfg.model.fine_matching.use_global_score,
            use_logits=cfg.model.fine_matching.use_logits,
            use_nms=cfg.model.fine_matching.use_nms,
            nms_radius=cfg.model.fine_matching.nms_radius,
            min_local_correspondences=cfg.model.fine_matching.min_local_correspondences,
            max_global_correspondences=cfg.model.fine_matching.max_global_correspondences,
            filter_correspondences=cfg.model.fine_matching.filter_correspondences,
            num_refinement_steps=cfg.model.fine_matching.num_refinement_steps,
        )

        self.dual_softmax = DualSoftmaxWithConfidence(
            cfg.model.dual_softmax.input_dim, cfg.model.dual_softmax.output_dim
        )

    def forward(self, data_dict: dict) -> dict:
        output_dict = {}

        dataset_name = data_dict["dataset"]
        scaling_factor = data_dict["scale"]

        matching_radius = self.matching_radius[dataset_name] * scaling_factor

        # Downsample point clouds
        feats = data_dict["feats"].detach()
        transform = data_dict["transform"].detach()

        src_length_c = data_dict["lengths"][-1][0].item()
        src_length_f = data_dict["lengths"][1][0].item()
        src_length = data_dict["lengths"][0][0].item()
        points_c = data_dict["points"][-1].detach()
        points_f = data_dict["points"][1].detach()
        points = data_dict["points"][0].detach()

        src_points_c = points_c[:src_length_c]
        tgt_points_c = points_c[src_length_c:]
        src_points_f = points_f[:src_length_f]
        tgt_points_f = points_f[src_length_f:]
        src_points = points[:src_length]
        tgt_points = points[src_length:]

        output_dict["src_points_c"] = src_points_c
        output_dict["tgt_points_c"] = tgt_points_c
        output_dict["src_points_f"] = src_points_f
        output_dict["tgt_points_f"] = tgt_points_f
        output_dict["src_points"] = src_points
        output_dict["tgt_points"] = tgt_points

        # 1. Generate ground truth node correspondences
        src_point_to_node_dict = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch, gather_points=True
        )
        tgt_point_to_node_dict = point_to_node_partition(
            tgt_points_f, tgt_points_c, self.num_points_in_patch, gather_points=True
        )
        src_node_masks = src_point_to_node_dict["node_masks"]
        src_node_knn_indices = src_point_to_node_dict["node_knn_indices"]
        src_node_knn_masks = src_point_to_node_dict["node_knn_masks"]
        tgt_node_masks = tgt_point_to_node_dict["node_masks"]
        tgt_node_knn_indices = tgt_point_to_node_dict["node_knn_indices"]
        tgt_node_knn_masks = tgt_point_to_node_dict["node_knn_masks"]

        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        tgt_padded_points_f = torch.cat([tgt_points_f, torch.zeros_like(tgt_points_f[:1])], dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)
        tgt_node_knn_points = index_select(tgt_padded_points_f, tgt_node_knn_indices, dim=0)

        # breakpoint()
        gt_src_node_corr_indices, gt_tgt_node_corr_indices, gt_node_corr_overlaps = get_patch_correspondences(
            src_points_c,
            tgt_points_c,
            src_node_knn_points,
            tgt_node_knn_points,
            transform,
            matching_radius,
            src_masks=src_node_masks,
            tgt_masks=tgt_node_masks,
            src_knn_masks=src_node_knn_masks,
            tgt_knn_masks=tgt_node_knn_masks,
        )

        output_dict["gt_src_node_corr_indices"] = gt_src_node_corr_indices
        output_dict["gt_tgt_node_corr_indices"] = gt_tgt_node_corr_indices
        output_dict["gt_node_corr_overlaps"] = gt_node_corr_overlaps

        torch.cuda.synchronize()
        start_time = time.time()

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        src_feats_c = feats_c[:src_length_c]
        tgt_feats_c = feats_c[src_length_c:]
        src_feats_c, tgt_feats_c = self.c_transformer(
            src_points_c.unsqueeze(0),
            tgt_points_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
            tgt_feats_c.unsqueeze(0),
        )
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        tgt_feats_c_norm = F.normalize(tgt_feats_c.squeeze(0), p=2, dim=1)

        output_dict["src_feats_c"] = src_feats_c_norm
        output_dict["tgt_feats_c"] = tgt_feats_c_norm

        # 5. Head for fine level matching
        src_feats_f = feats_f[:src_length_f]
        tgt_feats_f = feats_f[src_length_f:]
        output_dict["src_feats_f"] = src_feats_f
        output_dict["tgt_feats_f"] = tgt_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            src_node_corr_indices, tgt_node_corr_indices, node_corr_scores = self.coarse_matching(
                src_feats_c_norm, tgt_feats_c_norm, src_node_masks, tgt_node_masks
            )

            output_dict["src_node_corr_indices"] = src_node_corr_indices
            output_dict["tgt_node_corr_indices"] = tgt_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                src_node_corr_indices, tgt_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_src_node_corr_indices, gt_tgt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        tgt_node_corr_knn_indices = tgt_node_knn_indices[tgt_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        tgt_node_corr_knn_masks = tgt_node_knn_masks[tgt_node_corr_indices]  # (P, K)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)
        tgt_node_corr_knn_points = tgt_node_knn_points[tgt_node_corr_indices]  # (P, K, 3)

        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        tgt_padded_feats_f = torch.cat([tgt_feats_f, torch.zeros_like(tgt_feats_f[:1])], dim=0)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)
        tgt_node_corr_knn_feats = index_select(tgt_padded_feats_f, tgt_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict["src_node_corr_knn_indices"] = src_node_corr_knn_indices
        output_dict["tgt_node_corr_knn_indices"] = tgt_node_corr_knn_indices
        output_dict["src_node_corr_knn_points"] = src_node_corr_knn_points
        output_dict["tgt_node_corr_knn_points"] = tgt_node_corr_knn_points
        output_dict["src_node_corr_knn_masks"] = src_node_corr_knn_masks
        output_dict["tgt_node_corr_knn_masks"] = tgt_node_corr_knn_masks

        # 8. Fine Transformer
        src_node_corr_knn_feats, tgt_node_corr_knn_feats = self.f_transformer(
            src_node_corr_knn_points,
            tgt_node_corr_knn_points,
            src_node_corr_knn_feats,
            tgt_node_corr_knn_feats,
            src_masks=src_node_corr_knn_masks,
            tgt_masks=tgt_node_corr_knn_masks,
        )

        # 8. Optimal transport
        _, _, src_node_corr_knn_scores, tgt_node_corr_knn_scores, matching_scores = self.dual_softmax(
            src_node_corr_knn_feats, tgt_node_corr_knn_feats, src_node_corr_knn_masks, tgt_node_corr_knn_masks
        )

        output_dict["src_node_corr_knn_scores"] = src_node_corr_knn_scores
        output_dict["tgt_node_corr_knn_scores"] = tgt_node_corr_knn_scores
        output_dict["matching_scores"] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            src_corr_points, tgt_corr_points, corr_scores, estimated_transform = self.fine_matching(
                src_node_corr_knn_points,
                tgt_node_corr_knn_points,
                src_node_corr_knn_masks,
                tgt_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict["src_corr_points"] = src_corr_points
            output_dict["tgt_corr_points"] = tgt_corr_points
            output_dict["corr_scores"] = corr_scores
            output_dict["estimated_transform"] = estimated_transform

        # 10. some statistics
        # with torch.no_grad():
        #     aligned_src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)  # (B, K, 3)
        #     node_corr_knn_distance_mat = pairwise_distance(
        #         aligned_src_node_corr_knn_points, tgt_node_corr_knn_points, squared=False
        #     )  # (B, K, K)
        #     node_corr_knn_corr_mat = torch.lt(node_corr_knn_distance_mat, matching_radius)
        #     node_corr_knn_corr_mat = torch.logical_and(node_corr_knn_corr_mat, src_node_corr_knn_masks.unsqueeze(2))
        #     node_corr_knn_corr_mat = torch.logical_and(node_corr_knn_corr_mat, tgt_node_corr_knn_masks.unsqueeze(1))
        #     src_node_corr_knn_corr_masks = torch.any(node_corr_knn_corr_mat, dim=2)  # (B, K)
        #     tgt_node_corr_knn_corr_masks = torch.any(node_corr_knn_corr_mat, dim=1)  # (B, K)
        #     src_row_indices, src_col_indices = torch.nonzero(src_node_corr_knn_corr_masks, as_tuple=True)
        #     src_potential_indices = src_node_corr_knn_indices[src_row_indices, src_col_indices]
        #     src_potential_indices = torch.unique(src_potential_indices)
        #     tgt_row_indices, tgt_col_indices = torch.nonzero(tgt_node_corr_knn_corr_masks, as_tuple=True)
        #     tgt_potential_indices = tgt_node_corr_knn_indices[tgt_row_indices, tgt_col_indices]
        #     tgt_potential_indices = torch.unique(tgt_potential_indices)

        #     output_dict["src_potential_indices"] = src_potential_indices
        #     output_dict["tgt_potential_indices"] = tgt_potential_indices

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration

        return output_dict


def create_model(config: DictConfig) -> Module:
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    main()
