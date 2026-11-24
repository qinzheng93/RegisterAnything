from typing import Literal

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vision3d.array_ops.se3 import scale_transform
from vision3d.datasets.registration import (
    CollectedPairDataset,
    CSRegLidarPairDataset,
    CSRegSfmPairDataset,
    EthPairDataset,
    MitPairDataset,
    ModelNetPairDataset,
    NuScenesPairDataset,
    OdometryKittiPairDataset,
    OxfordPairDataset,
    ScanNetPPiPhonePairDataset,
    ShapeNetPairDataset,
    ThreeDMatchPairDataset,
    TiersPairDataset,
    WodPairDataset,
)
from vision3d.utils.collate import RegistrationCollateFnPackModeWithGraphPyramid
from vision3d.utils.dataloader import build_dataloader, calibrate_neighbors_pack_mode


def get_dataset_cfg(dataset_name: str, cfg: DictConfig) -> DictConfig:
    match dataset_name:
        case "3DMatch":
            return cfg.data.threedmatch
        case "odometryKitti":
            return cfg.data.kitti
        case "ModelNet":
            return cfg.data.modelnet
        case "nuScenes":
            return cfg.data.nuscenes
        case "nuScenes":
            return cfg.data.eth
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_all_dataset_cfgs(cfg: DictConfig, phase: Literal["train", "test"]) -> list[dict]:
    dataset_names = cfg.data.dataset_names

    all_cfgs = []
    for dataset_name in dataset_names:
        cur_cfg = {}
        cur_cfg["name"] = dataset_name
        dataset_cfg = get_dataset_cfg(dataset_name, cfg)
        cur_cfg["dataset_dir"] = dataset_cfg.dataset_dir
        kwargs = {}
        if "base" in dataset_cfg:
            kwargs.update(dataset_cfg["base"])
        kwargs.update(dataset_cfg[phase])
        cur_cfg["kwargs"] = kwargs
        all_cfgs.append(cur_cfg)

    return all_cfgs


class TransformFn:
    def __init__(self, cfg: DictConfig, phase: Literal["train", "test"], voxel_size_0: float = 0.025):
        self.voxel_size_0 = voxel_size_0
        if phase == "train":
            self.voxel_size_dict = {
                "3DMatch": 0.025,
                "odometryKitti": 0.3,
                "nuScenes": 0.3,
                "ModelNet": self.voxel_size_0,
            }
        else:
            self.voxel_size_dict = {
                "3DMatch": cfg.data.threedmatch.test.voxel_size,
                "ScanNet++i": cfg.data.scannetppi.voxel_size,
                "3DCSR-L": cfg.data.csreg_l.voxel_size,
                "3DCSR-S": cfg.data.csreg_s.voxel_size,
                "odometryKitti": cfg.data.kitti.test.voxel_size,
                "nuScenes": cfg.data.nuscenes.test.voxel_size,
                "ETH": cfg.data.eth.voxel_size,
                "TIERS": cfg.data.tiers.voxel_size,
                "MIT": cfg.data.mit.voxel_size,
                "WOD": cfg.data.wod.voxel_size,
                "Oxford": cfg.data.oxford.voxel_size,
                "ModelNet": self.voxel_size_0,
                "ShapeNet": self.voxel_size_0,
            }

    def __call__(self, data_dict: dict) -> dict:
        src_points = data_dict["src_points"]
        tgt_points = data_dict["tgt_points"]
        transform = data_dict["transform"]
        # compose packed points
        dataset_name = data_dict["dataset"]
        if "voxel_size" in data_dict:
            voxel_size_1 = data_dict["voxel_size"]
        else:
            voxel_size_1 = self.voxel_size_dict[dataset_name]
        scale = self.voxel_size_0 / voxel_size_1
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


def train_valid_data_loader(cfg: DictConfig) -> tuple[DataLoader, DataLoader, list[int]]:
    train_cfgs = get_all_dataset_cfgs(cfg, "train")
    test_cfgs = get_all_dataset_cfgs(cfg, "test")

    train_transform = TransformFn(cfg, "train")
    test_transform = TransformFn(cfg, "test")
    for train_cfg in train_cfgs:
        train_cfg["kwargs"]["transform_fn"] = train_transform
    for test_cfg in test_cfgs:
        test_cfg["kwargs"]["transform_fn"] = test_transform

    train_dataset = CollectedPairDataset("train", train_cfgs)

    collate_fn = RegistrationCollateFnPackModeWithGraphPyramid(
        cfg.model.backbone.num_stages, cfg.model.backbone.base_voxel_size, cfg.model.backbone.base_radius
    )

    neighbor_limits = calibrate_neighbors_pack_mode(train_dataset, collate_fn)

    collate_fn = RegistrationCollateFnPackModeWithGraphPyramid(
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_radius,
        neighbor_limits=neighbor_limits,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataset = CollectedPairDataset("val", test_cfgs)

    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, neighbor_limits


def compute_neighbor_limits(cfg: DictConfig) -> list[int]:
    train_cfgs = get_all_dataset_cfgs(cfg, "train")

    transform_fn = TransformFn(cfg, "train")
    for train_cfg in train_cfgs:
        train_cfg["kwargs"]["transform_fn"] = transform_fn

    train_dataset = CollectedPairDataset("train", train_cfgs)

    collate_fn = RegistrationCollateFnPackModeWithGraphPyramid(
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_radius,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(train_dataset, collate_fn)

    return neighbor_limits


def test_data_loader(cfg: DictConfig, benchmark: str) -> tuple[DataLoader, list[int]]:
    neighbor_limits = compute_neighbor_limits(cfg)

    collate_fn = RegistrationCollateFnPackModeWithGraphPyramid(
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_radius,
        neighbor_limits=neighbor_limits,
    )

    transform_fn = TransformFn(cfg, "test")

    match benchmark:
        case "3DMatch" | "3DLoMatch":
            test_dataset = ThreeDMatchPairDataset(
                cfg.data.threedmatch.dataset_dir,
                benchmark,
                transform_fn=transform_fn,
                max_points=cfg.data.threedmatch.test.max_points,
                use_augmentation=False,
                use_raw_data=cfg.data.threedmatch.test.use_raw_data,
                voxel_size=cfg.data.threedmatch.test.voxel_size,
                geometry_norm=cfg.data.threedmatch.test.geometry_norm,
                rotated=cfg.data.threedmatch.test.rotated,
            )
        case "ScanNet++i":
            test_dataset = ScanNetPPiPhonePairDataset(
                cfg.data.scannetppi.dataset_dir,
                subset=cfg.data.scannetppi.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.scannetppi.max_points,
                voxel_size=cfg.data.scannetppi.voxel_size,
                geometry_norm=cfg.data.scannetppi.geometry_norm,
            )
        case "3DCSR-L":
            test_dataset = CSRegLidarPairDataset(
                cfg.data.csreg_l.dataset_dir,
                subset=cfg.data.csreg_l.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.csreg_l.max_points,
                voxel_size=cfg.data.csreg_l.voxel_size,
                geometry_norm=cfg.data.csreg_l.geometry_norm,
            )
        case "3DCSR-S":
            test_dataset = CSRegSfmPairDataset(
                cfg.data.csreg_s.dataset_dir,
                subset=cfg.data.csreg_s.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.csreg_s.max_points,
                voxel_size=cfg.data.csreg_s.voxel_size,
                geometry_norm=cfg.data.csreg_s.geometry_norm,
            )
        case "odometryKitti":
            test_dataset = OdometryKittiPairDataset(
                cfg.data.kitti.dataset_dir,
                "test",
                transform_fn=transform_fn,
                max_points=cfg.data.kitti.test.max_points,
                use_augmentation=False,
                use_raw_data=cfg.data.kitti.test.use_raw_data,
                voxel_size=cfg.data.kitti.test.voxel_size,
                geometry_norm=cfg.data.kitti.test.geometry_norm,
            )
        case "nuScenes":
            test_dataset = NuScenesPairDataset(
                cfg.data.nuscenes.dataset_dir,
                "test",
                transform_fn=transform_fn,
                max_points=cfg.data.nuscenes.test.max_points,
                use_augmentation=False,
                use_raw_data=cfg.data.nuscenes.test.use_raw_data,
                voxel_size=cfg.data.nuscenes.test.voxel_size,
                geometry_norm=cfg.data.nuscenes.test.geometry_norm,
                max_distance=cfg.data.nuscenes.test.max_distance,
            )
        case "ETH":
            test_dataset = EthPairDataset(
                cfg.data.eth.dataset_dir,
                subset=cfg.data.eth.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.eth.max_points,
                voxel_size=cfg.data.eth.voxel_size,
                geometry_norm=cfg.data.eth.geometry_norm,
            )
        case "TIERS":
            test_dataset = TiersPairDataset(
                cfg.data.tiers.dataset_dir,
                subset=cfg.data.tiers.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.tiers.max_points,
                voxel_size=cfg.data.tiers.voxel_size,
                geometry_norm=cfg.data.tiers.geometry_norm,
            )
        case "MIT":
            test_dataset = MitPairDataset(
                cfg.data.mit.dataset_dir,
                subset=cfg.data.mit.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.mit.max_points,
                voxel_size=cfg.data.mit.voxel_size,
                geometry_norm=cfg.data.mit.geometry_norm,
            )
        case "WOD":
            test_dataset = WodPairDataset(
                cfg.data.wod.dataset_dir,
                subset=cfg.data.wod.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.wod.max_points,
                voxel_size=cfg.data.wod.voxel_size,
                geometry_norm=cfg.data.wod.geometry_norm,
            )
        case "Oxford":
            test_dataset = OxfordPairDataset(
                cfg.data.oxford.dataset_dir,
                subset=cfg.data.oxford.subset,
                transform_fn=transform_fn,
                max_points=cfg.data.oxford.max_points,
                voxel_size=cfg.data.oxford.voxel_size,
                geometry_norm=cfg.data.oxford.geometry_norm,
            )
        case "ModelNet":
            test_dataset = ModelNetPairDataset(
                cfg.data.modelnet.dataset_dir,
                "test",
                transform_fn=transform_fn,
                num_points=cfg.data.modelnet.base.num_points,
                voxel_size=cfg.data.modelnet.base.voxel_size,
                rotation_magnitude=cfg.data.modelnet.base.rotation_magnitude,
                translation_magnitude=cfg.data.modelnet.base.translation_magnitude,
                noise_magnitude=cfg.data.modelnet.test.noise_magnitude,
                keep_ratio=cfg.data.modelnet.base.keep_ratio,
                crop_method=cfg.data.modelnet.base.crop_method,
                asymmetric=cfg.data.modelnet.base.asymmetric,
                class_indices=cfg.data.modelnet.test.class_indices,
                deterministic=True,
                twice_sample=cfg.data.modelnet.base.twice_sample,
                twice_transform=cfg.data.modelnet.base.twice_transform,
                return_normals=False,
                return_occupancy=True,
            )
        case "ShapeNet":
            test_dataset = ShapeNetPairDataset(
                cfg.data.shapenet.dataset_dir,
                cfg.data.shapenet.test.subset,
                transform_fn=transform_fn,
                num_points=cfg.data.shapenet.base.num_points,
                voxel_size=cfg.data.shapenet.test.voxel_size,
                rotation_magnitude=cfg.data.shapenet.base.rotation_magnitude,
                translation_magnitude=cfg.data.shapenet.base.translation_magnitude,
                noise_magnitude=cfg.data.shapenet.test.noise_magnitude,
                keep_ratio=cfg.data.shapenet.base.keep_ratio,
                crop_method=cfg.data.shapenet.base.crop_method,
                deterministic=True,
                twice_sample=cfg.data.shapenet.base.twice_sample,
                twice_transform=cfg.data.shapenet.base.twice_transform,
                return_normals=False,
                return_occupancy=True,
                geometry_norm=cfg.data.shapenet.test.geometry_norm,
            )
        case _:
            raise ValueError(f"Unknown dataset name: {benchmark}")

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return test_loader, neighbor_limits
