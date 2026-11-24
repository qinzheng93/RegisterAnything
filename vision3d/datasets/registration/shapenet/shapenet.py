import os.path as osp
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from vision3d.array_ops.point_cloud_utils import (
    normalize_points,
    random_crop_points_from_viewpoint,
    random_crop_points_with_plane,
    random_jitter_points,
    random_sample_points_with_padding,
    random_sample_viewpoint,
    random_shuffle_points,
    regularize_normals,
)
from vision3d.array_ops.registration_utils import get_correspondences, point_cloud_overlap
from vision3d.array_ops.se3 import apply_transform, inverse_transform, random_sample_transform
from vision3d.models.register_anything.scale_norm import geometry_aware_normalization
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import estimate_normals, voxel_down_sample
from vision3d.utils.types import TransformFn


class ShapeNetPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: TransformFn | None = None,
        num_points: int = 1024,
        voxel_size: float | None = None,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: float | None = None,
        keep_ratio: float = 0.7,
        crop_method: Literal["plane", "point"] = "plane",
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = False,
        return_occupancy: bool = False,
        return_corr_indices: bool = False,
        matching_radius: float = 0.05,
        min_overlap: float = 0.1,
        max_overlap: float | None = None,
        use_random_scaling: bool = False,
        augmentation_min_scale: float = 0.8,
        augmentation_max_scale: float = 1.2,
        use_estimated_normals: bool = False,
        overfitting_index: int | None = None,
        geometry_norm: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "seen", "unseen", "test"]
        assert crop_method in ["plane", "point"]

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.transform_fn = transform_fn

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.use_random_scaling = use_random_scaling
        self.aug_min_scale = augmentation_min_scale
        self.aug_max_scale = augmentation_max_scale
        self.use_estimated_normals = use_estimated_normals
        self.overfitting_index = overfitting_index
        self.geometry_norm = geometry_norm

        self.data_list = load_pickle(osp.join(dataset_dir, "metadata", f"{subset}.pkl"))

    def __getitem__(self, index: int) -> dict:
        if self.overfitting_index is not None:
            index = self.overfitting_index

        metadata_dict: dict = self.data_list[index]
        data_dict = np.load(osp.join(self.dataset_dir, "data", metadata_dict["model_path"]))
        raw_points = data_dict["points"].copy()
        raw_normals = data_dict["normals"].copy()

        # NOTE: specific for ShapeNet
        # swap axes to make (x->front, y->right, z->up) from (x->left, y->up, z->back)
        raw_points = np.stack([-raw_points[:, 2], -raw_points[:, 0], raw_points[:, 1]], axis=1)
        raw_normals = np.stack([-raw_normals[:, 2], -raw_normals[:, 0], raw_normals[:, 1]], axis=1)

        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # normalize raw point cloud
        raw_points = normalize_points(raw_points)

        # random scaling
        if self.use_random_scaling:
            scale = np.random.uniform(self.aug_min_scale, self.aug_max_scale)
            raw_points = raw_points * scale

        # once sample on raw point cloud
        if not self.twice_sample:
            raw_points, raw_normals = random_sample_points_with_padding(
                raw_points, self.num_points, normals=raw_normals
            )

        # split target and source point cloud
        tgt_points = raw_points.copy()
        tgt_normals = raw_normals.copy()

        # twice transform
        if self.twice_transform:
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            tgt_points, tgt_normals = apply_transform(tgt_points, transform, normals=tgt_normals)

        src_points = tgt_points.copy()
        src_normals = tgt_normals.copy()

        # random transform to source point cloud
        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        inv_transform = inverse_transform(transform)
        src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

        raw_src_points = src_points
        raw_src_normals = src_normals
        raw_tgt_points = tgt_points
        raw_tgt_normals = tgt_normals

        while True:
            src_points = raw_src_points
            src_normals = raw_src_normals
            tgt_points = raw_tgt_points
            tgt_normals = raw_tgt_normals
            # crop
            if self.keep_ratio is not None:
                if self.crop_method == "plane":
                    src_points, src_normals = random_crop_points_with_plane(
                        src_points, normals=src_normals, keep_ratio=self.keep_ratio
                    )
                    tgt_points, tgt_normals = random_crop_points_with_plane(
                        tgt_points, normals=tgt_normals, keep_ratio=self.keep_ratio
                    )
                else:
                    viewpoint = random_sample_viewpoint()
                    src_points, src_normals = random_crop_points_from_viewpoint(
                        src_points, normals=src_normals, viewpoint=viewpoint, keep_ratio=self.keep_ratio
                    )
                    tgt_points, tgt_normals = random_crop_points_from_viewpoint(
                        tgt_points, normals=tgt_normals, viewpoint=viewpoint, keep_ratio=self.keep_ratio
                    )

            # check overlap
            is_available = True
            overlap = point_cloud_overlap(src_points, tgt_points, transform, positive_radius=0.05)
            is_available = is_available and overlap >= self.min_overlap
            if self.max_overlap is not None:
                is_available = is_available and overlap <= self.max_overlap
            if is_available:
                break

        if self.twice_sample:
            # twice sample on both point clouds
            src_points, src_normals = random_sample_points_with_padding(
                src_points, self.num_points, normals=src_normals
            )
            tgt_points, tgt_normals = random_sample_points_with_padding(
                tgt_points, self.num_points, normals=tgt_normals
            )

        # random jitter
        if self.noise_magnitude is not None:
            src_points = random_jitter_points(src_points, sigma=0.01, scale=self.noise_magnitude)
            tgt_points = random_jitter_points(tgt_points, sigma=0.01, scale=self.noise_magnitude)

        # random shuffle
        src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)
        tgt_points, tgt_normals = random_shuffle_points(tgt_points, normals=tgt_normals)

        # voxelization
        voxel_size = self.voxel_size
        if self.geometry_norm:
            voxel_size = geometry_aware_normalization(src_points, tgt_points)
        if voxel_size is not None:
            src_points, src_normals = voxel_down_sample(src_points, voxel_size, normals=src_normals)
            tgt_points, tgt_normals = voxel_down_sample(tgt_points, voxel_size, normals=tgt_normals)

        new_data_dict = {
            "dataset": "ShapeNet",
            "category": metadata_dict["category"],
            "model_id": metadata_dict["model_id"],
            "raw_points": raw_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "tgt_points": tgt_points.astype(np.float32),
            "transform": transform.astype(np.float32),
            "index": int(index),
        }

        if self.use_estimated_normals:
            tgt_normals = estimate_normals(tgt_points)
            tgt_normals = regularize_normals(tgt_points, tgt_normals)
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)

        if self.return_normals:
            new_data_dict["raw_normals"] = raw_normals.astype(np.float32)
            new_data_dict["src_normals"] = src_normals.astype(np.float32)
            new_data_dict["tgt_normals"] = tgt_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict["src_feats"] = np.ones_like(src_points[:, :1]).astype(np.float32)
            new_data_dict["tgt_feats"] = np.ones_like(tgt_points[:, :1]).astype(np.float32)

        if self.return_corr_indices:
            src_corr_indices, tgt_corr_indices = get_correspondences(
                src_points, tgt_points, transform, self.matching_radius
            )
            new_data_dict["src_corr_indices"] = src_corr_indices
            new_data_dict["tgt_corr_indices"] = tgt_corr_indices

        if self.transform_fn is not None:
            new_data_dict = self.transform_fn(new_data_dict)

        return new_data_dict

    def __len__(self) -> int:
        return len(self.data_list)
