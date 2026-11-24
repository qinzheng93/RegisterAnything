import json
import os
import os.path as osp
import time
from argparse import ArgumentParser, Namespace

import numpy as np
from omegaconf import DictConfig, OmegaConf

from vision3d.array_ops.metrics import isotropic_registration_error
from vision3d.array_ops.procrustes import weighted_procrustes
from vision3d.array_ops.registration_utils import evaluate_correspondences, evaluate_sparse_correspondences
from vision3d.utils.logger import logger
from vision3d.utils.open3d import registration_with_ransac_from_correspondences
from vision3d.utils.parser import get_default_parser
from vision3d.utils.summary_board import SummaryBoard

# isort: split
from config import make_cfg


def make_parser() -> ArgumentParser:
    parser = get_default_parser()
    parser.add_argument(
        "--benchmark",
        choices=[
            "3DMatch",
            "3DLoMatch",
            "ScanNet++i",
            "odometryKitti",
            "nuScenes",
            "ETH",
            "TIERS",
            "MIT",
            "WOD",
            "Oxford",
        ],
        help="test benchmark",
    )
    parser.add_argument("--test_epoch", type=int, required=True, help="test epoch")
    parser.add_argument("--method", choices=["lgr", "ransac", "svd"], required=True, help="registration method")
    parser.add_argument("--num_corrs", type=int, default=None, help="number of correspondences for registration")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    return parser


def eval_one_epoch(args: Namespace, cfg: DictConfig) -> None:
    benchmark = args.benchmark
    cache_dir = osp.join(cfg.exp.cache_dir, benchmark)

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter("precision")
    coarse_matching_meter.register_meter("PMR>0")
    coarse_matching_meter.register_meter("PMR>=0.1")
    coarse_matching_meter.register_meter("PMR>=0.3")
    coarse_matching_meter.register_meter("PMR>=0.5")

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter("recall")
    fine_matching_meter.register_meter("inlier_ratio")
    fine_matching_meter.register_meter("overlap")

    registration_meter = SummaryBoard()
    registration_meter.register_meter("recall")
    registration_meter.register_meter("rre")
    registration_meter.register_meter("rte")

    acceptance_radius = cfg.eval.acceptance_radius[benchmark]
    inlier_ratio_threshold = cfg.eval.inlier_ratio_threshold
    rre_threshold = cfg.eval.rre_threshold[benchmark]
    rte_threshold = cfg.eval.rte_threshold[benchmark]

    scene_names = sorted(os.listdir(cache_dir))
    for scene_name in scene_names:
        scene_dir = osp.join(cache_dir, scene_name)
        filenames = sorted(os.listdir(scene_dir))

        num_test_pairs = len(filenames)
        for i, filename in enumerate(filenames):
            src_frame, tgt_frame = osp.splitext(filename)[0].split("-")

            data_dict = np.load(osp.join(scene_dir, filename))

            tgt_nodes = data_dict["tgt_points_c"]
            src_nodes = data_dict["src_points_c"]
            tgt_node_corr_indices = data_dict["tgt_node_corr_indices"]
            src_node_corr_indices = data_dict["src_node_corr_indices"]

            src_corr_points = data_dict["src_corr_points"]
            tgt_corr_points = data_dict["tgt_corr_points"]
            corr_scores = data_dict["corr_scores"]

            gt_src_node_corr_indices = data_dict["gt_src_node_corr_indices"]
            gt_tgt_node_corr_indices = data_dict["gt_tgt_node_corr_indices"]
            gt_transform = data_dict["transform"]

            if args.num_corrs is not None and corr_scores.shape[0] > args.num_corrs:
                sel_indices = np.argsort(-corr_scores)[: args.num_corrs]
                src_corr_points = src_corr_points[sel_indices]
                tgt_corr_points = tgt_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]

            messages = []
            messages.append(f"{i + 1}/{num_test_pairs}")
            messages.append(f"{scene_name}")
            messages.append(f"src: {src_frame}")
            messages.append(f"tgt: {tgt_frame}")

            # 1. evaluate correspondences
            # 1.1 evaluate coarse correspondences
            coarse_matching_result_dict = evaluate_sparse_correspondences(
                src_nodes.shape[0],
                tgt_nodes.shape[0],
                src_node_corr_indices,
                tgt_node_corr_indices,
                gt_src_node_corr_indices,
                gt_tgt_node_corr_indices,
            )

            coarse_precision = coarse_matching_result_dict["precision"]

            coarse_matching_meter.update("precision", coarse_precision)
            coarse_matching_meter.update("PMR>0", float(coarse_precision > 0))
            coarse_matching_meter.update("PMR>=0.1", float(coarse_precision >= 0.1))
            coarse_matching_meter.update("PMR>=0.3", float(coarse_precision >= 0.3))
            coarse_matching_meter.update("PMR>=0.5", float(coarse_precision >= 0.5))

            # 1.2 evaluate fine correspondences
            fine_matching_result_dict = evaluate_correspondences(
                src_corr_points, tgt_corr_points, gt_transform, positive_radius=acceptance_radius
            )

            inlier_ratio = fine_matching_result_dict["inlier_ratio"]
            overlap = fine_matching_result_dict["overlap"]

            fine_matching_meter.update("inlier_ratio", inlier_ratio)
            fine_matching_meter.update("overlap", overlap)
            fine_matching_meter.update("recall", float(inlier_ratio >= inlier_ratio_threshold))

            messages.append(f"c_PIR: {coarse_precision:.3f}")
            messages.append(f"f_IR: {inlier_ratio:.3f}")
            messages.append(f"f_OV: {overlap:.3f}")
            messages.append(f"f_CD: {fine_matching_result_dict['distance']:.3f}")
            messages.append(f"f_NU: {src_corr_points.shape[0]}")

            # 2. evaluate registration
            match args.method:
                case "lgr":
                    est_transform = data_dict["estimated_transform"]
                case "ransac":
                    est_transform = registration_with_ransac_from_correspondences(
                        src_corr_points,
                        tgt_corr_points,
                        distance_threshold=cfg.ransac.distance_threshold[benchmark],
                        ransac_n=cfg.ransac.num_points[benchmark],
                        num_iterations=cfg.ransac.num_iterations,
                    )
                case "svd":
                    est_transform = weighted_procrustes(src_corr_points, tgt_corr_points, corr_scores)
                case _:
                    raise ValueError(f"Unsupported registration method: {args.method}.")

            rre, rte = isotropic_registration_error(gt_transform, est_transform)
            accepted = rre < rre_threshold and rte < rte_threshold
            if accepted:
                registration_meter.update("rre", rre)
                registration_meter.update("rte", rte)
            registration_meter.update("recall", float(accepted))
            messages.append(f"r_RRE: {rre:.3f}")
            messages.append(f"r_RTE: {rte:.3f}")

            if args.verbose:
                message = ", ".join(messages)
                logger.info(message)

    logger.success(f"Epoch {args.test_epoch}")

    # 1. print correspondence evaluation results
    messages = []
    messages.append("  Coarse Matching")
    messages.append(f"PIR: {coarse_matching_meter.mean('precision'):.3f}")
    messages.append(f"PMR>0: {coarse_matching_meter.mean('PMR>0'):.3f}")
    messages.append(f"PMR>=0.1: {coarse_matching_meter.mean('PMR>=0.1'):.3f}")
    messages.append(f"PMR>=0.3: {coarse_matching_meter.mean('PMR>=0.3'):.3f}")
    messages.append(f"PMR>=0.5: {coarse_matching_meter.mean('PMR>=0.5'):.3f}")
    message = ", ".join(messages)
    logger.success(message)

    messages = []
    messages.append("  Fine Matching")
    messages.append(f"FMR: {fine_matching_meter.mean('recall'):.3f}")
    messages.append(f"IR: {fine_matching_meter.mean('inlier_ratio'):.3f}")
    messages.append(f"OV: {fine_matching_meter.mean('overlap'):.3f}")
    messages.append(f"std: {fine_matching_meter.std('recall'):.3f}")
    message = ", ".join(messages)
    logger.success(message)

    # 2. print registration evaluation results
    messages = []
    messages.append("  Registration")
    messages.append(f"RR: {registration_meter.mean('recall'):.3f}")
    messages.append(f"RRE: {registration_meter.mean('rre'):.3f}")
    messages.append(f"RTE: {registration_meter.mean('rte'):.3f}")
    message = ", ".join(messages)
    logger.success(message)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.exp.log_dir, "eval-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
    logger.initialize(log_file=log_file)

    message = "Configs:\n" + json.dumps(OmegaConf.to_object(cfg), indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg)


if __name__ == "__main__":
    main()
