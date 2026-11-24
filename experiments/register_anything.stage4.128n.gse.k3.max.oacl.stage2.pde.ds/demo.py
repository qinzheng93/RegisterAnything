import argparse

import numpy as np
import open3d as o3d
import torch

from vision3d.array_ops.se3 import scale_transform
from vision3d.engine.checkpoint_manager import load_state_dict
from vision3d.models.register_anything.scale_norm import geometry_aware_normalization
from vision3d.utils.collate import RegistrationCollateFnPackModeWithGraphPyramid
from vision3d.utils.open3d import (
    draw_geometries,
    force_x11_backend,
    get_color,
    make_open3d_colors,
    make_open3d_point_cloud,
    voxel_down_sample,
)
from vision3d.utils.tensor import move_to_cuda, tensor_to_array

# isort: split
from config import make_cfg
from loss import EvalFunction
from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--src_pcd", type=str, required=True, help="Path to the source point cloud")
    parser.add_argument("--tgt_pcd", type=str, required=True, help="Path to the target point cloud")
    parser.add_argument("--transform", type=str, required=True, help="Path to the transform file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results")
    args = parser.parse_args()
    return args


def transform_fn(data_dict: dict, voxel_size_0: float = 0.025) -> dict:
    src_points = data_dict["src_points"]
    tgt_points = data_dict["tgt_points"]
    transform = data_dict["transform"]
    scale = voxel_size_0 / data_dict["voxel_size"]
    # scale points and transform
    src_points = src_points * scale
    tgt_points = tgt_points * scale
    transform = scale_transform(transform, scale)
    # update dict
    data_dict["src_points"] = src_points.astype(np.float32)
    data_dict["tgt_points"] = tgt_points.astype(np.float32)
    data_dict["transform"] = transform.astype(np.float32)
    data_dict["scale"] = scale
    return data_dict


def main():
    args = parse_args()
    cfg = make_cfg()

    # load model
    model = create_model(cfg)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    load_state_dict(model, state_dict["model"], strict=True)
    model.cuda()
    model.eval()

    eval_func = EvalFunction(cfg)

    # load data
    src_pcd = o3d.io.read_point_cloud(args.src_pcd)
    tgt_pcd = o3d.io.read_point_cloud(args.tgt_pcd)
    transform = np.loadtxt(args.transform)

    src_points = np.asarray(src_pcd.points)
    tgt_points = np.asarray(tgt_pcd.points)

    voxel_size = geometry_aware_normalization(src_points, tgt_points, min_voxel_size=0.015)
    src_points = voxel_down_sample(src_points, voxel_size)
    tgt_points = voxel_down_sample(tgt_points, voxel_size)

    data_dict = {
        "dataset": "ScanNet++i",
        "src_points": src_points.astype(np.float32),
        "tgt_points": tgt_points.astype(np.float32),
        "src_feats": np.ones(shape=(src_points.shape[0], 1), dtype=np.float32),
        "tgt_feats": np.ones(shape=(tgt_points.shape[0], 1), dtype=np.float32),
        "transform": transform.astype(np.float32),
        "voxel_size": voxel_size,
    }
    data_dict = transform_fn(data_dict)

    collate_fn = RegistrationCollateFnPackModeWithGraphPyramid(
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_radius,
        neighbor_limits=[32, 32, 32, 32],
    )
    data_dict = collate_fn([data_dict])
    data_dict = move_to_cuda(data_dict)

    # run model
    output_dict = model(data_dict)
    result_dict = eval_func(data_dict, output_dict)

    print(result_dict)

    output_dict = tensor_to_array(output_dict)

    if args.output_path is not None:
        np.save(args.output_path, output_dict)

    # visualize
    if not args.visualize:
        return

    force_x11_backend()

    src_color = get_color("custom_blue")
    tgt_color = get_color("custom_yellow")

    src_colors = make_open3d_colors(src_points, src_color)
    tgt_colors = make_open3d_colors(tgt_points, tgt_color)
    src_pcd = make_open3d_point_cloud(src_points, src_colors)
    tgt_pcd = make_open3d_point_cloud(tgt_points, tgt_colors)
    src_pcd.estimate_normals()
    tgt_pcd.estimate_normals()
    src_pcd.orient_normals_towards_camera_location()
    tgt_pcd.orient_normals_towards_camera_location()

    draw_geometries(src_pcd)
    draw_geometries(tgt_pcd)

    estimate_transform = output_dict["estimated_transform"]
    src_pcd.transform(estimate_transform)
    draw_geometries(src_pcd, tgt_pcd)


if __name__ == "__main__":
    main()
