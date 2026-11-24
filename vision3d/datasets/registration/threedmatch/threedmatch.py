import os.path as osp
import random

import numpy as np
import open3d as o3d
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.point_cloud_utils import random_crop_points_with_plane, random_sample_points
from vision3d.array_ops.registration_utils import get_correspondences, point_cloud_overlap
from vision3d.array_ops.se3 import (
    apply_transform,
    compose_transforms,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
    scale_transform,
)
from vision3d.array_ops.so3 import apply_rotation, random_sample_rotation
from vision3d.models.register_anything.scale_norm import geometry_aware_normalization
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import voxel_down_sample
from vision3d.utils.types import TransformFn


class ThreeDMatchPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: TransformFn | None = None,
        max_points: int | None = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1,
        use_weak_augmentation: bool = False,
        use_random_scaling: bool = False,
        augmentation_min_scale: float = 0.8,
        augmentation_max_scale: float = 1.2,
        use_random_crop: bool = False,
        keep_ratio: float = 0.7,
        overlap_threshold: float | None = None,
        return_corr_indices: bool = False,
        matching_radius: float | None = None,
        rotated: bool = False,
        deterministic: bool = False,
        scaling_factor: float | None = None,
        use_raw_data: bool = False,
        voxel_size: float = 0.025,
        geometry_norm: bool = False,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.subset = subset

        self.transform_fn = transform_fn
        self.max_points = max_points
        self.use_aug = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.use_weak_aug = use_weak_augmentation
        self.use_random_scaling = use_random_scaling
        self.aug_min_scale = augmentation_min_scale
        self.aug_max_scale = augmentation_max_scale
        self.use_random_crop = use_random_crop
        self.keep_ratio = keep_ratio
        self.overlap_threshold = overlap_threshold
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        self.rotated = rotated
        self.deterministic = deterministic
        self.scaling_factor = scaling_factor
        self.use_raw_data = use_raw_data
        self.voxel_size = voxel_size
        self.geometry_norm = geometry_norm

        if self.return_corr_indices:
            assert self.matching_radius is not None, "'matching_radius' is None but 'return_corr_indices' is True."

        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{subset}.pkl"))

        if self.overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] > self.overlap_threshold]

    def __len__(self) -> int:
        return len(self.metadata_list)

    def load_point_cloud(self, file_name: str) -> ndarray:
        """Load point cloud.
        Random sample if there are too many points.
        NOTE: setting "max_points" with "num_workers" > 1 will cause nondeterminism.
        """
        if self.use_raw_data:
            file_name = file_name.replace(".pth", ".ply")
            pcd = o3d.io.read_point_cloud(osp.join(self.data_dir, file_name))
            points = np.asarray(pcd.points)
        else:
            points = torch.load(osp.join(self.data_dir, file_name), weights_only=False)
        return points

    def weakly_augment_point_cloud(
        self, src_points: ndarray, tgt_points: ndarray, transform: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Weakly augment point clouds as proposed in RegTR.

        tgt_points = src_points @ rotation.T + translation

        1. Randomly sample a rotation axis and a small rotation angle.
        2. Randomly sample a small translation.
        3. Move point clouds to the original point.
        4. Apply random augmentation transformation.
        5. Move back to their initial positions.
        6. Random noise.
        """
        aug_transform = random_sample_small_transform()
        if random.random() > 0.5:
            tgt_center = tgt_points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -tgt_center)
            add_center = get_transform_from_rotation_translation(None, tgt_center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            tgt_points = apply_transform(tgt_points, aug_transform)
            transform = compose_transforms(transform, aug_transform)
        else:
            src_center = src_points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -src_center)
            add_center = get_transform_from_rotation_translation(None, src_center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            src_points = apply_transform(src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise

        return src_points, tgt_points, transform

    def augment_point_cloud(
        self, src_points: ndarray, tgt_points: ndarray, transform: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Augment point clouds.

        tgt_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_transform = get_transform_from_rotation_translation(aug_rotation, None)
        if random.random() > 0.5:
            tgt_points = apply_transform(tgt_points, aug_transform)
            transform = compose_transforms(transform, aug_transform)
        else:
            src_points = apply_transform(src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise

        return src_points, tgt_points, transform

    def random_crop(
        self, src_points: ndarray, tgt_points: ndarray, transform: ndarray, overlap: float, max_attempts: int = 3
    ) -> tuple[ndarray, ndarray, float]:
        """Random crop point clouds."""
        if overlap <= 0.3:
            return src_points, tgt_points, overlap
        for _ in range(max_attempts):
            new_src_points = random_crop_points_with_plane(src_points, keep_ratio=self.keep_ratio)
            new_tgt_points = random_crop_points_with_plane(tgt_points, keep_ratio=self.keep_ratio)
            new_overlap = point_cloud_overlap(new_src_points, new_tgt_points, transform=transform, positive_radius=0.1)
            if new_overlap > 0.1:
                return new_src_points, new_tgt_points, new_overlap
        return src_points, tgt_points, overlap

    def __getitem__(self, index: int) -> dict:
        # deterministic
        if self.deterministic:
            np.random.seed(index)

        data_dict = {}
        data_dict["dataset"] = "3DMatch"

        # metadata
        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["src_frame"] = metadata["src_frame"]
        data_dict["tgt_frame"] = metadata["tgt_frame"]

        # get overlap
        overlap = metadata["overlap"]

        # get transformation
        transform = metadata["transform"]

        # get point cloud
        src_points = self.load_point_cloud(metadata["src_path"])
        tgt_points = self.load_point_cloud(metadata["tgt_path"])

        # voxelize
        if self.use_raw_data:
            # voxelize for raw data
            voxel_size = self.voxel_size
            if self.geometry_norm:
                voxel_size = geometry_aware_normalization(src_points, tgt_points)
            if voxel_size is not None:
                src_points = voxel_down_sample(src_points, voxel_size)
                tgt_points = voxel_down_sample(tgt_points, voxel_size)
                data_dict["voxel_size"] = voxel_size

        # random sample
        if self.max_points is not None:
            src_points = random_sample_points(src_points, self.max_points)
            tgt_points = random_sample_points(tgt_points, self.max_points)

        # random scaling
        if self.use_random_scaling:
            scale = np.random.uniform(self.aug_min_scale, self.aug_max_scale)
            src_points = src_points * scale
            tgt_points = tgt_points * scale
            transform = scale_transform(transform, scale)

        # augmentation
        if self.use_aug:
            if self.use_weak_aug:
                src_points, tgt_points, transform = self.weakly_augment_point_cloud(src_points, tgt_points, transform)
            else:
                src_points, tgt_points, transform = self.augment_point_cloud(src_points, tgt_points, transform)

        # random crop
        if self.use_random_crop:
            src_points, tgt_points, overlap = self.random_crop(src_points, tgt_points, transform, overlap)

        # apply random rotation to both point clouds
        if self.rotated:
            # random rotate src point cloud
            src_rotation = random_sample_rotation()
            src_points = apply_rotation(src_points, src_rotation)

            # random rotate tgt point cloud
            tgt_rotation = random_sample_rotation()
            tgt_points = apply_rotation(tgt_points, tgt_rotation)

            # adjust ground-truth transformation
            src_transform = get_transform_from_rotation_translation(src_rotation, None)
            inv_src_transform = inverse_transform(src_transform)
            tgt_transform = get_transform_from_rotation_translation(tgt_rotation, None)
            transform = compose_transforms(inv_src_transform, transform, tgt_transform)

        # get correspondences
        if self.return_corr_indices:
            assert self.matching_radius is not None, "return_corr_indices is True but matching_radius is None."
            src_corr_indices, tgt_corr_indices = get_correspondences(
                src_points, tgt_points, transform, self.matching_radius
            )
            data_dict["src_corr_indices"] = src_corr_indices
            data_dict["tgt_corr_indices"] = tgt_corr_indices

        if self.scaling_factor is not None:
            src_points = src_points * self.scaling_factor
            tgt_points = tgt_points * self.scaling_factor
            transform = scale_transform(transform, self.scaling_factor)
            data_dict["scaling_factor"] = self.scaling_factor

        data_dict["src_points"] = src_points.astype(np.float32)
        data_dict["tgt_points"] = tgt_points.astype(np.float32)
        data_dict["src_feats"] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict["tgt_feats"] = np.ones((tgt_points.shape[0], 1), dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["overlap"] = overlap

        # post transform fn
        if self.transform_fn is not None:
            data_dict = self.transform_fn(data_dict)

        return data_dict
