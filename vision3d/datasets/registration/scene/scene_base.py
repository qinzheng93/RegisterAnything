import os.path as osp

import numpy as np
import open3d as o3d
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.point_cloud_utils import random_sample_points
from vision3d.array_ops.registration_utils import get_correspondences
from vision3d.array_ops.se3 import get_rotation_translation_from_transform, get_transform_from_rotation_translation
from vision3d.models.register_anything.scale_norm import geometry_aware_normalization
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import voxel_down_sample
from vision3d.utils.types import TransformFn


class ScenePairDatasetBase(Dataset):
    dataset_name: str

    def __init__(
        self,
        dataset_dir: str,
        subset: str = "test",
        transform_fn: TransformFn | None = None,
        max_points: int | None = None,
        return_corr_indices: bool = False,
        matching_radius: float | None = None,
        scaling_factor: float | None = None,
        voxel_size: float | None = None,
        min_range: float | None = None,
        max_range: float | None = None,
        max_distance: float | None = None,
        geometry_norm: bool = False,
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
        self.geometry_norm = geometry_norm

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError("`matching_radius` is None but `return_corr_indices` is set.")

        self.metadata = load_pickle(osp.join(self.dataset_dir, "metadata", f"{subset}.pkl"))

    def load_point_cloud(self, pcd_path: str) -> ndarray:
        pcd = o3d.io.read_point_cloud(osp.join(self.dataset_dir, "data", pcd_path))
        points = np.asarray(pcd.points)
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

    def generate_data(self, data_dict: dict, metadata: dict) -> dict:
        # load point cloud
        src_points = self.load_point_cloud(metadata["src_pcd"])
        tgt_points = self.load_point_cloud(metadata["tgt_pcd"])
        transform = metadata["transform"]

        # voxelization
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

        # add correspondences
        if self.return_corr_indices:
            if self.matching_radius is None:
                raise ValueError("`matching_radius` is None but `return_corr_indices` is set.")
            src_corr_indices, tgt_corr_indices = get_correspondences(
                src_points, tgt_points, transform, self.matching_radius
            )
            data_dict["src_corr_indices"] = src_corr_indices
            data_dict["tgt_corr_indices"] = tgt_corr_indices

        # apply scaling
        if self.scaling_factor is not None:
            src_points = src_points * self.scaling_factor
            tgt_points = tgt_points * self.scaling_factor
            rotation, translation = get_rotation_translation_from_transform(transform)
            translation = translation * self.scaling_factor
            transform = get_transform_from_rotation_translation(rotation, translation)
            data_dict["scaling_factor"] = self.scaling_factor

        # update dict
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
