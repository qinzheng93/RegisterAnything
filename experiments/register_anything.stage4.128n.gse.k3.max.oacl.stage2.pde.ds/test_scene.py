import os.path as osp

import numpy as np
from omegaconf import DictConfig

from vision3d.engine import SampleTester
from vision3d.ops.se3 import scale_transform
from vision3d.utils.io import ensure_dir
from vision3d.utils.logger import logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args, get_default_parser
from vision3d.utils.profiling import profile_cpu_runtime
from vision3d.utils.tensor import tensor_to_array

# isort: split
from config import make_cfg
from dataset import test_data_loader
from loss import EvalFunction
from model import create_model


def add_custom_args():
    parser = get_default_parser()
    parser.add_argument_group("experiment", "experiment arguments")
    parser.add_argument(
        "--benchmark",
        choices=[
            "3DMatch",
            "3DLoMatch",
            "ScanNet++i",
            "3DCSR-L",
            "3DCSR-S",
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
    parser.add_argument("--save_features", action="store_true", help="save features")


class Tester(SampleTester):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader create"):
            data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        logger.info(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.eval_func = EvalFunction(cfg)

        # preparation
        self.output_dir = osp.join(cfg.exp.cache_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

        self.rre_list = []
        self.rte_list = []
        self.cache_dir = cfg.exp.cache_dir

    def test_step(self, iteration: int, data_dict: dict) -> dict:
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration: int, data_dict: dict, output_dict: dict) -> dict:
        result_dict = self.eval_func(data_dict, output_dict)
        result_dict["duration"] = output_dict["duration"]
        return result_dict

    def get_log_string(self, iteration: int, data_dict: dict, output_dict: dict, result_dict: dict) -> str:
        messages = []
        messages.append(f"{data_dict['scene_name']}")
        if data_dict["dataset"] == "nuScenes":
            # In nuScenes, limit the token id to 8 characters
            messages.append(f"src: {data_dict['src_frame'][:8]}")
            messages.append(f"tgt: {data_dict['tgt_frame'][:8]}")
        else:
            messages.append(f"src: {data_dict['src_frame']}")
            messages.append(f"tgt: {data_dict['tgt_frame']}")
        messages.append(get_log_string(result_dict=result_dict))
        messages.append(f"nCorr: {output_dict['corr_scores'].shape[0]}")
        messages.append(f"scale: {data_dict['scale']:.3f}")
        if "voxel_size" in data_dict:
            messages.append(f"voxel_size: {data_dict['voxel_size']:g}")
        message = ", ".join(messages)
        return message

    def after_test_step(self, iteration: int, data_dict: dict, output_dict: dict, result_dict: dict) -> None:
        scene_name = data_dict["scene_name"]
        src_frame = data_dict["src_frame"]
        tgt_frame = data_dict["tgt_frame"]

        scaling_factor = 1.0 / data_dict["scale"]
        src_points = output_dict["src_points"] * scaling_factor
        tgt_points = output_dict["tgt_points"] * scaling_factor
        src_points_f = output_dict["src_points_f"] * scaling_factor
        tgt_points_f = output_dict["tgt_points_f"] * scaling_factor
        src_points_c = output_dict["src_points_c"] * scaling_factor
        tgt_points_c = output_dict["tgt_points_c"] * scaling_factor
        src_corr_points = output_dict["src_corr_points"] * scaling_factor
        tgt_corr_points = output_dict["tgt_corr_points"] * scaling_factor
        estimated_transform = scale_transform(output_dict["estimated_transform"], scaling_factor)
        transform = scale_transform(data_dict["transform"], scaling_factor)

        save_dict = {}

        save_dict["src_points"] = tensor_to_array(src_points)
        save_dict["tgt_points"] = tensor_to_array(tgt_points)
        save_dict["src_points_f"] = tensor_to_array(src_points_f)
        save_dict["tgt_points_f"] = tensor_to_array(tgt_points_f)
        save_dict["src_points_c"] = tensor_to_array(src_points_c)
        save_dict["tgt_points_c"] = tensor_to_array(tgt_points_c)

        if self.args.save_features:
            save_dict["src_feats_c"] = tensor_to_array(output_dict["src_feats_c"])
            save_dict["tgt_feats_c"] = tensor_to_array(output_dict["tgt_feats_c"])

        save_dict["src_node_corr_indices"] = tensor_to_array(output_dict["src_node_corr_indices"])
        save_dict["tgt_node_corr_indices"] = tensor_to_array(output_dict["tgt_node_corr_indices"])
        save_dict["src_corr_points"] = tensor_to_array(src_corr_points)
        save_dict["tgt_corr_points"] = tensor_to_array(tgt_corr_points)
        save_dict["corr_scores"] = tensor_to_array(output_dict["corr_scores"])
        save_dict["gt_src_node_corr_indices"] = tensor_to_array(output_dict["gt_src_node_corr_indices"])
        save_dict["gt_tgt_node_corr_indices"] = tensor_to_array(output_dict["gt_tgt_node_corr_indices"])
        save_dict["gt_node_corr_overlaps"] = tensor_to_array(output_dict["gt_node_corr_overlaps"])

        save_dict["estimated_transform"] = tensor_to_array(estimated_transform)
        save_dict["transform"] = tensor_to_array(transform)

        if "overlap" in data_dict:
            save_dict["overlap"] = data_dict["overlap"]

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f"{src_frame}-{tgt_frame}.npz")
        np.savez_compressed(file_name, **save_dict)

        self.rre_list.append(tensor_to_array(result_dict["RRE"]))
        self.rte_list.append(tensor_to_array(result_dict["RTE"]))

    def after_test_epoch(self, summary_dict: dict) -> None:
        file_name = osp.join(self.cache_dir, f"{self.args.benchmark}.npz")
        rre = np.asarray(self.rre_list)
        rte = np.asarray(self.rte_list)
        np.savez_compressed(file_name, rre=rre, rte=rte)


def main():
    add_tester_args()
    add_custom_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
