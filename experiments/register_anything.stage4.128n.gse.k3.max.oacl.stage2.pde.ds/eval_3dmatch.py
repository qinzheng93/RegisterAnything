import glob
import json
import os.path as osp
import time
from argparse import ArgumentParser, Namespace

import numpy as np
from omegaconf import DictConfig, OmegaConf

from vision3d.array_ops.metrics import isotropic_registration_error
from vision3d.array_ops.procrustes import weighted_procrustes
from vision3d.array_ops.registration_utils import evaluate_correspondences, evaluate_sparse_correspondences
from vision3d.datasets.registration.threedmatch.threedmatch_utils import (
    compute_transform_error,
    get_gt_logs_and_infos,
    get_num_fragments,
    get_scene_abbr,
    write_log_file,
)
from vision3d.utils.logger import logger
from vision3d.utils.open3d import registration_with_ransac_from_correspondences
from vision3d.utils.parser import get_default_parser
from vision3d.utils.summary_board import SummaryBoard

# isort: split
from config import make_cfg


def make_parser() -> ArgumentParser:
    parser = get_default_parser()
    parser.add_argument("--test_epoch", type=int, required=True, help="test epoch")
    parser.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], required=True, help="test benchmark")
    parser.add_argument("--method", choices=["lgr", "ransac", "svd"], required=True, help="registration method")
    parser.add_argument("--num_corrs", type=int, default=None, help="number of correspondences for registration")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    return parser


def eval_one_epoch(args: Namespace, cfg: DictConfig, log_file: str) -> None:
    features_dir = osp.join(cfg.exp.cache_dir, args.benchmark)
    benchmark = args.benchmark

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter("precision")
    coarse_matching_meter.register_meter("PMR>0")
    coarse_matching_meter.register_meter("PMR>=0.1")
    coarse_matching_meter.register_meter("PMR>=0.3")
    coarse_matching_meter.register_meter("PMR>=0.5")
    coarse_matching_meter.register_meter("scene_precision")
    coarse_matching_meter.register_meter("scene_PMR>0")
    coarse_matching_meter.register_meter("scene_PMR>=0.1")
    coarse_matching_meter.register_meter("scene_PMR>=0.3")
    coarse_matching_meter.register_meter("scene_PMR>=0.5")

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter("recall")
    fine_matching_meter.register_meter("inlier_ratio")
    fine_matching_meter.register_meter("overlap")
    fine_matching_meter.register_meter("scene_recall")
    fine_matching_meter.register_meter("scene_inlier_ratio")
    fine_matching_meter.register_meter("scene_overlap")

    registration_meter = SummaryBoard()
    registration_meter.register_meter("recall")
    registration_meter.register_meter("mean_rre")
    registration_meter.register_meter("mean_rte")
    registration_meter.register_meter("median_rre")
    registration_meter.register_meter("median_rte")
    registration_meter.register_meter("scene_recall")
    registration_meter.register_meter("scene_rre")
    registration_meter.register_meter("scene_rte")

    scene_coarse_matching_result_dict = {}
    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    acceptance_radius = cfg.eval.acceptance_radius["3DMatch"]
    inlier_ratio_threshold = cfg.eval.inlier_ratio_threshold
    rmse_threshold = cfg.eval.rmse_threshold["3DMatch"]

    scene_dirs = sorted(glob.glob(osp.join(features_dir, "*")))
    for scene_dir in scene_dirs:
        coarse_matching_meter.reset_meter("scene_precision")
        coarse_matching_meter.reset_meter("scene_PMR>0")
        coarse_matching_meter.reset_meter("scene_PMR>=0.1")
        coarse_matching_meter.reset_meter("scene_PMR>=0.3")
        coarse_matching_meter.reset_meter("scene_PMR>=0.5")

        fine_matching_meter.reset_meter("scene_recall")
        fine_matching_meter.reset_meter("scene_inlier_ratio")
        fine_matching_meter.reset_meter("scene_overlap")

        registration_meter.reset_meter("scene_recall")
        registration_meter.reset_meter("scene_rre")
        registration_meter.reset_meter("scene_rte")

        scene_name = osp.basename(scene_dir)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)
        gt_dir = osp.join(cfg.data.threedmatch.dataset_dir, "metadata", "benchmarks", benchmark, scene_name)
        gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_dir, num_fragments)

        estimated_transforms = []

        file_names = sorted(
            glob.glob(osp.join(scene_dir, "*.npz")),
            key=lambda x: [int(i) for i in osp.basename(x).split(".")[0].split("-")],
        )
        for file_name in file_names:
            src_frame, tgt_frame = [int(x) for x in osp.basename(file_name).split(".")[0].split("-")]

            data_dict = np.load(file_name)

            src_points_c = data_dict["src_points_c"]
            tgt_points_c = data_dict["tgt_points_c"]
            src_node_corr_indices = data_dict["src_node_corr_indices"]
            tgt_node_corr_indices = data_dict["tgt_node_corr_indices"]

            src_corr_points = data_dict["src_corr_points"]
            tgt_corr_points = data_dict["tgt_corr_points"]
            corr_scores = data_dict["corr_scores"]

            gt_src_node_corr_indices = data_dict["gt_src_node_corr_indices"]
            gt_tgt_node_corr_indices = data_dict["gt_tgt_node_corr_indices"]
            transform = data_dict["transform"]
            pcd_overlap = data_dict["overlap"]

            if args.num_corrs is not None and corr_scores.shape[0] > args.num_corrs:
                sel_indices = np.argsort(-corr_scores)[: args.num_corrs]
                src_corr_points = src_corr_points[sel_indices]
                tgt_corr_points = tgt_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]

            messages = []
            messages.append(f"{scene_abbr}")
            messages.append(f"src: {src_frame}")
            messages.append(f"tgt: {tgt_frame}")
            messages.append(f"OV: {pcd_overlap:.3f}")

            # 1. evaluate correspondences
            # 1.1 evaluate coarse correspondences
            coarse_matching_result_dict = evaluate_sparse_correspondences(
                src_points_c.shape[0],
                tgt_points_c.shape[0],
                src_node_corr_indices,
                tgt_node_corr_indices,
                gt_src_node_corr_indices,
                gt_tgt_node_corr_indices,
            )

            coarse_precision = coarse_matching_result_dict["precision"]

            coarse_matching_meter.update("scene_precision", coarse_precision)
            coarse_matching_meter.update("scene_PMR>0", float(coarse_precision > 0))
            coarse_matching_meter.update("scene_PMR>=0.1", float(coarse_precision >= 0.1))
            coarse_matching_meter.update("scene_PMR>=0.3", float(coarse_precision >= 0.3))
            coarse_matching_meter.update("scene_PMR>=0.5", float(coarse_precision >= 0.5))

            # 1.2 evaluate fine correspondences
            fine_matching_result_dict = evaluate_correspondences(
                src_corr_points, tgt_corr_points, transform, positive_radius=acceptance_radius
            )

            inlier_ratio = fine_matching_result_dict["inlier_ratio"]
            overlap = fine_matching_result_dict["overlap"]

            fine_matching_meter.update("scene_inlier_ratio", inlier_ratio)
            fine_matching_meter.update("scene_overlap", overlap)
            fine_matching_meter.update("scene_recall", float(inlier_ratio >= inlier_ratio_threshold))

            messages.append(f"c_PIR: {coarse_precision:.3f}")
            messages.append(f"f_IR: {inlier_ratio:.3f}")
            messages.append(f"f_OV: {overlap:.3f}")
            messages.append(f"f_RS: {fine_matching_result_dict['distance']:.3f}")
            messages.append(f"f_NU: {corr_scores.shape[0]}")

            # 2. evaluate registration
            match args.method:
                case "lgr":
                    estimated_transform = data_dict["estimated_transform"]
                case "ransac":
                    estimated_transform = registration_with_ransac_from_correspondences(
                        src_corr_points,
                        tgt_corr_points,
                        distance_threshold=cfg.ransac.distance_threshold["3DMatch"],
                        ransac_n=cfg.ransac.num_points["3DMatch"],
                        num_iterations=cfg.ransac.num_iterations,
                    )
                case "svd":
                    estimated_transform = weighted_procrustes(src_corr_points, tgt_corr_points, corr_scores)
                case _:
                    raise ValueError(f"Unsupported registration method: {args.method}.")

            estimated_transforms.append(
                {
                    "test_pair": [tgt_frame, src_frame],
                    "num_fragments": num_fragments,
                    "transform": estimated_transform,
                }
            )

            if gt_indices[tgt_frame, src_frame] != -1:
                # evaluate transform (realignment error)
                gt_index = gt_indices[tgt_frame, src_frame]
                transform = gt_logs[gt_index]["transform"]
                covariance = gt_infos[gt_index]["covariance"]
                error = compute_transform_error(transform, covariance, estimated_transform)
                messages.append(f"r_RMSE: {np.sqrt(error):.3f}")
                accepted = error < rmse_threshold**2
                registration_meter.update("scene_recall", float(accepted))
                if accepted:
                    rre, rte = isotropic_registration_error(transform, estimated_transform)
                    registration_meter.update("scene_rre", rre)
                    registration_meter.update("scene_rte", rte)
                    messages.append(f"r_RRE: {rre:.3f}")
                    messages.append(f"r_RTE: {rte:.3f}")

            # Evaluate re-alignment error
            # if tgt_frame + 1 < src_frame:
            #     evaluate transform (realignment error)
            #     src_points_f = data_dict['src_points_f']
            #     error = compute_realignment_error(src_points_f, transform, estimated_transform)
            #     messages.append(f"r_RMSE: {error:.3f}")
            #     accepted = error < config.eval_rmse_threshold
            #     registration_meter.update('scene_recall', float(accepted))
            #     if accepted:
            #         rre, rte = isotropic_registration_error(transform, estimated_transform)
            #         registration_meter.update('scene_rre', rre)
            #         registration_meter.update('scene_rte', rte)
            #         messages.append(f"r_RRE: {rre:.3f}")
            #         messages.append(f"r_RTE: {rte:.3f}")

            if args.verbose:
                message = ", ".join(messages)
                logger.info(message)

        est_log = osp.join(cfg.exp.result_dir, benchmark, scene_name, "est.log")
        write_log_file(est_log, estimated_transforms)

        logger.info(f"Scene_name: {scene_name}")

        # 1. print correspondence evaluation results (one scene)
        # 1.1 coarse level statistics
        coarse_precision = coarse_matching_meter.mean("scene_precision")
        coarse_matching_recall_0 = coarse_matching_meter.mean("scene_PMR>0")
        coarse_matching_recall_1 = coarse_matching_meter.mean("scene_PMR>=0.1")
        coarse_matching_recall_3 = coarse_matching_meter.mean("scene_PMR>=0.3")
        coarse_matching_recall_5 = coarse_matching_meter.mean("scene_PMR>=0.5")
        coarse_matching_meter.update("precision", coarse_precision)
        coarse_matching_meter.update("PMR>0", coarse_matching_recall_0)
        coarse_matching_meter.update("PMR>=0.1", coarse_matching_recall_1)
        coarse_matching_meter.update("PMR>=0.3", coarse_matching_recall_3)
        coarse_matching_meter.update("PMR>=0.5", coarse_matching_recall_5)
        scene_coarse_matching_result_dict[scene_abbr] = {
            "precision": coarse_precision,
            "PMR>0": coarse_matching_recall_0,
            "PMR>=0.1": coarse_matching_recall_1,
            "PMR>=0.3": coarse_matching_recall_3,
            "PMR>=0.5": coarse_matching_recall_5,
        }

        # 1.2 fine level statistics
        recall = fine_matching_meter.mean("scene_recall")
        inlier_ratio = fine_matching_meter.mean("scene_inlier_ratio")
        overlap = fine_matching_meter.mean("scene_overlap")
        fine_matching_meter.update("recall", recall)
        fine_matching_meter.update("inlier_ratio", inlier_ratio)
        fine_matching_meter.update("overlap", overlap)
        scene_fine_matching_result_dict[scene_abbr] = {"recall": recall, "inlier_ratio": inlier_ratio}

        messages = []
        messages.append("  Correspondence")
        messages.append(f"c_PIR: {coarse_precision:.3f}")
        messages.append(f"c_PMR>0: {coarse_matching_recall_0:.3f}")
        messages.append(f"c_PMR>=0.1: {coarse_matching_recall_1:.3f}")
        messages.append(f"c_PMR>=0.3: {coarse_matching_recall_3:.3f}")
        messages.append(f"c_PMR>=0.5: {coarse_matching_recall_5:.3f}")
        messages.append(f"f_FMR: {recall:.3f}")
        messages.append(f"f_IR: {inlier_ratio:.3f}")
        messages.append(f"f_OV: {overlap:.3f}")
        message = ", ".join(messages)
        logger.info(message)

        # 2. print registration evaluation results (one scene)
        recall = registration_meter.mean("scene_recall")
        mean_rre = registration_meter.mean("scene_rre")
        mean_rte = registration_meter.mean("scene_rte")
        median_rre = registration_meter.median("scene_rre")
        median_rte = registration_meter.median("scene_rte")
        registration_meter.update("recall", recall)
        registration_meter.update("mean_rre", mean_rre)
        registration_meter.update("mean_rte", mean_rte)
        registration_meter.update("median_rre", median_rre)
        registration_meter.update("median_rte", median_rte)

        scene_registration_result_dict[scene_abbr] = {
            "recall": recall,
            "mean_rre": mean_rre,
            "mean_rte": mean_rte,
            "median_rre": median_rre,
            "median_rte": median_rte,
        }

        messages = []
        messages.append("  Registration")
        messages.append(f"RR: {recall:.3f}")
        messages.append(f"mean_RRE: {mean_rre:.3f}")
        messages.append(f"mean_RTE: {mean_rte:.3f}")
        messages.append(f"median_RRE: {median_rre:.3f}")
        messages.append(f"median_RTE: {median_rte:.3f}")
        message = ", ".join(messages)
        logger.info(message)

    logger.success("Epoch {}".format(args.test_epoch))

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
    for scene_abbr, result_dict in scene_coarse_matching_result_dict.items():
        messages = []
        messages.append(f"    {scene_abbr}")
        messages.append(f"PIR: {result_dict['precision']:.3f}")
        messages.append(f"PMR>0: {result_dict['PMR>0']:.3f}")
        messages.append(f"PMR>=0.1: {result_dict['PMR>=0.1']:.3f}")
        messages.append(f"PMR>=0.3: {result_dict['PMR>=0.3']:.3f}")
        messages.append(f"PMR>=0.5: {result_dict['PMR>=0.5']:.3f}")
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
    for scene_abbr, result_dict in scene_fine_matching_result_dict.items():
        messages = []
        messages.append(f"    {scene_abbr}")
        messages.append(f"FMR: {result_dict['recall']:.3f}")
        messages.append(f"IR: {result_dict['inlier_ratio']:.3f}")
        message = ", ".join(messages)
        logger.success(message)

    # 2. print registration evaluation results
    messages = []
    messages.append("  Registration")
    messages.append(f"RR: {registration_meter.mean('recall'):.3f}")
    messages.append(f"mean_RRE: {registration_meter.mean('mean_rre'):.3f}")
    messages.append(f"mean_RTE: {registration_meter.mean('mean_rte'):.3f}")
    messages.append(f"median_RRE: {registration_meter.mean('median_rre'):.3f}")
    messages.append(f"median_RTE: {registration_meter.mean('median_rte'):.3f}")
    message = ", ".join(messages)
    logger.success(message)
    for scene_abbr, result_dict in scene_registration_result_dict.items():
        messages = []
        messages.append(f"    {scene_abbr}")
        messages.append(f"RR: {result_dict['recall']:.3f}")
        messages.append(f"mean_RRE: {result_dict['mean_rre']:.3f}")
        messages.append(f"mean_RTE: {result_dict['mean_rte']:.3f}")
        messages.append(f"median_RRE: {result_dict['median_rre']:.3f}")
        messages.append(f"median_RTE: {result_dict['median_rte']:.3f}")
        message = ", ".join(messages)
        logger.success(message)

    summary_dict = {
        "log_file": osp.basename(log_file),
        "epoch": args.test_epoch,
        "PIR": coarse_matching_meter.mean("precision"),
        "FMR": fine_matching_meter.mean("recall"),
        "IR": fine_matching_meter.mean("inlier_ratio"),
        "RR": registration_meter.mean("recall"),
    }
    with open(osp.join(cfg.exp.log_dir, args.benchmark + ".json"), "a") as f:
        result_item = json.dumps(summary_dict) + "\n"
        f.write(result_item)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.exp.log_dir, "eval-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
    logger.initialize(log_file=log_file)

    message = "Configs:\n" + json.dumps(OmegaConf.to_object(cfg), indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, log_file)


if __name__ == "__main__":
    main()
