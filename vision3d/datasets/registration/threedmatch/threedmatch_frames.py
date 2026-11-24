import os.path as osp
import random

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.registration_utils import get_correspondences
from vision3d.array_ops.se3 import (
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
    inverse_transform,
)
from vision3d.array_ops.so3 import random_sample_rotation, random_sample_rotation_norm
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import make_open3d_point_cloud
from vision3d.utils.types import TransformFn


class ThreeDMatchFramesPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform: TransformFn | None = None,
        point_limit: int | None = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1,
        overlap_threshold: float | None = None,
        return_corr_indices: bool = False,
        matching_radius: float | None = None,
        rotated: bool = False,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.data_dir = osp.join(self.dataset_dir, "data")

        self.transform = transform
        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"positive_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{subset}.pkl"))

        if self.overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] > self.overlap_threshold]

    def __len__(self) -> int:
        return len(self.metadata_list)

    def load_point_cloud_and_pose(self, file_name: str) -> tuple[ndarray, ndarray]:
        data = np.load(osp.join(self.data_dir, file_name))
        points = data["points"]
        pose = data["pose"]
        pcd = make_open3d_point_cloud(points)
        pcd = pcd.voxel_down_sample(0.025)
        points = np.asarray(pcd.points)
        # NOTE: setting "max_points" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points, pose

    def augment_point_cloud(
        self, ref_points: ndarray, src_points: ndarray, rotation: ndarray, translation: ndarray
    ) -> tuple[ndarray, ...]:
        """Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index: int) -> dict:
        data_dict = {}

        # metadata
        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["ref_frame"] = metadata["tgt_frame"][-4:]
        data_dict["src_frame"] = metadata["src_frame"][-4:]
        data_dict["overlap"] = metadata["overlap"]

        # get point cloud
        ref_points, ref_pose = self.load_point_cloud_and_pose(metadata["tgt_file"])
        src_points, src_pose = self.load_point_cloud_and_pose(metadata["src_file"])
        transform = np.matmul(inverse_transform(ref_pose), src_pose)
        rotation, translation = get_rotation_translation_from_transform(transform)

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self.augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = random_sample_rotation_norm()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_norm()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            assert self.matching_radius is not None, "return_corr_indices is True but matching_radius is None."
            corr_indices = get_correspondences(src_points, ref_points, transform, self.matching_radius)
            data_dict["corr_indices"] = corr_indices

        data_dict["ref_points"] = ref_points.astype(np.float32)
        data_dict["src_points"] = src_points.astype(np.float32)
        data_dict["ref_feats"] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict["src_feats"] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
