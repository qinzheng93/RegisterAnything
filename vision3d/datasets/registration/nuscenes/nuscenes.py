import os.path as osp
import random

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.point_cloud_utils import random_crop_points_with_plane, random_sample_points
from vision3d.array_ops.registration_utils import get_correspondences, point_cloud_overlap
from vision3d.array_ops.se3 import (
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
    scale_transform,
)
from vision3d.array_ops.so3 import random_sample_rotation
from vision3d.models.register_anything.scale_norm import geometry_aware_normalization
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import voxel_down_sample
from vision3d.utils.types import TransformFn


class NuScenesPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: TransformFn | None = None,
        max_points: int | None = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1.0,
        augmentation_shift: float = 2.0,
        use_random_scaling: bool = True,
        augmentation_min_scale: float = 0.8,
        augmentation_max_scale: float = 1.2,
        use_random_crop: bool = False,
        keep_ratio: float = 0.7,
        return_corr_indices: bool = False,
        matching_radius: float | None = None,
        scaling_factor: float | None = None,
        voxel_size: float = 0.3,
        geometry_norm: bool = False,
        min_range: float | None = None,
        max_range: float | None = None,
        max_distance: float | None = None,
        use_raw_data: bool = True,
    ):
        super().__init__()

        self.transform_fn = transform_fn

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.max_points = max_points

        self.scaling_factor = scaling_factor
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range
        self.max_distance = max_distance
        self.use_raw_data = use_raw_data
        self.geometry_norm = geometry_norm

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.use_random_scaling = use_random_scaling
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale

        self.use_random_crop = use_random_crop
        self.keep_ratio = keep_ratio

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"positive_radius" is None but "return_corr_indices" is set.')

        self.metadata = load_pickle(osp.join(self.dataset_dir, "metadata", f"{subset}.pkl"))

    def augment_point_cloud(
        self, src_points: ndarray, tgt_points: ndarray, transform: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        tgt_points = tgt_points + (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            tgt_points = np.matmul(tgt_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random shift
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        tgt_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_points = src_points + src_shift
        tgt_points = tgt_points + tgt_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + tgt_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return src_points, tgt_points, transform

    def load_point_cloud(self, file_name: str) -> ndarray:
        if self.use_raw_data:
            points: ndarray = np.load(osp.join(self.dataset_dir, "data", file_name))
        else:
            points: ndarray = np.load(osp.join(self.dataset_dir, "downsampled", file_name))
        if self.min_range is not None:
            masks = np.amin(points, axis=1) >= self.min_range
            points = points[masks]
        if self.max_range is not None:
            masks = np.amax(points, axis=1) <= self.max_range
            points = points[masks]
        if self.max_distance is not None:
            distances = np.linalg.norm(points, axis=1)
            masks = distances <= self.max_distance
            points = points[masks]
        return points

    def random_crop(
        self, src_points: ndarray, tgt_points: ndarray, transform: ndarray, overlap: float, max_attempts: int = 3
    ) -> tuple[ndarray, ndarray, float]:
        """Random crop point clouds."""
        if overlap < 0.4:
            return src_points, tgt_points, overlap
        for _ in range(max_attempts):
            new_src_points = random_crop_points_with_plane(src_points, keep_ratio=self.keep_ratio)
            new_tgt_points = random_crop_points_with_plane(tgt_points, keep_ratio=self.keep_ratio)
            new_overlap = point_cloud_overlap(new_src_points, new_tgt_points, transform=transform, positive_radius=0.6)
            if new_overlap > 0.3:
                return new_src_points, new_tgt_points, new_overlap
        return src_points, tgt_points, overlap

    def __getitem__(self, index: int) -> dict:
        data_dict = {}
        data_dict["dataset"] = "nuScenes"

        metadata: dict = self.metadata[index]
        data_dict["scene_name"] = metadata["scene_token"]
        data_dict["src_frame"] = metadata["src_token"]
        data_dict["tgt_frame"] = metadata["tgt_token"]

        src_points = self.load_point_cloud(metadata["src_file"])
        tgt_points = self.load_point_cloud(metadata["tgt_file"])
        transform: ndarray = metadata["transform"]

        # voxelize
        if self.use_raw_data:
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

        overlap = point_cloud_overlap(src_points, tgt_points, transform=transform, positive_radius=0.6)

        # random scaling
        if self.use_random_scaling:
            scale = np.random.uniform(self.augmentation_min_scale, self.augmentation_max_scale)
            src_points = src_points * scale
            tgt_points = tgt_points * scale
            transform = scale_transform(transform, scale)

        # augmentation
        if self.use_augmentation:
            src_points, tgt_points, transform = self.augment_point_cloud(src_points, tgt_points, transform)

        # random crop
        if self.use_random_crop:
            src_points, tgt_points, overlap = self.random_crop(src_points, tgt_points, transform, overlap)

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
        data_dict["src_feats"] = np.ones(shape=(src_points.shape[0], 1), dtype=np.float32)
        data_dict["tgt_feats"] = np.ones(shape=(tgt_points.shape[0], 1), dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)

        # post transform fn
        if self.transform_fn is not None:
            data_dict = self.transform_fn(data_dict)

        return data_dict

    def __len__(self) -> int:
        return len(self.metadata)
