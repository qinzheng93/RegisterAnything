import os.path as osp

import numpy as np
import skimage.io as io
import torch
from torch.utils.data import Dataset

from vision3d.array_ops.registration_utils import get_2d3d_correspondences_mutual
from vision3d.array_ops.se3 import (
    apply_transform,
    compose_transforms,
    get_transform_from_rotation_translation,
    inverse_transform,
)
from vision3d.array_ops.so3 import random_sample_rotation
from vision3d.utils.io import load_pickle
from vision3d.utils.types import TransformFn


class TwoDThreeDMatchPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: TransformFn | None = None,
        max_points: int | None = None,
        return_corr_indices: bool = False,
        matching_radius_3d: float = 0.015,
        matching_radius_2d: float = 12.0,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1.0,
        overlap_threshold: float | None = None,
        use_gray_image: bool = True,
        scene_names: list[str] | None = None,
    ):
        super().__init__()

        assert subset in ["train", "val", "3DMatch"]

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset

        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_radius_3d = matching_radius_3d
        self.matching_radius_2d = matching_radius_2d
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.overlap_threshold = overlap_threshold
        self.use_gray_image = use_gray_image
        self.scene_names = scene_names

        self.transform_fn = transform_fn

        self.metadata_list: list[dict] = load_pickle(osp.join(self.metadata_dir, f"{self.subset}-2d3d.pkl"))
        if self.overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] > self.overlap_threshold]
        if self.scene_names is not None:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] in self.scene_names]

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, index: int) -> dict:
        data_dict = {}

        index = index % len(self.metadata_list)
        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["img_frame"] = metadata["img_frame"]
        data_dict["pcd_frame"] = metadata["pcd_frame"]
        data_dict["img_overlap"] = metadata["img_overlap"]
        data_dict["pcd_overlap"] = metadata["pcd_overlap"]
        data_dict["overlap"] = metadata["overlap"]
        data_dict["img_overlap"] = metadata["img_overlap"]
        data_dict["pcd_overlap"] = metadata["pcd_overlap"]

        intrinsics = metadata["intrinsics"]
        transform = metadata["transform"]  # point cloud to image

        # read image
        depth = io.imread(osp.join(self.data_dir, metadata["img_depth_path"]))
        image = io.imread(osp.join(self.data_dir, metadata["img_color_path"]), as_gray=self.use_gray_image)
        if self.use_gray_image:
            image -= image.mean()
        else:
            image = image / 255.0  # normalize to [0, 1]
            image -= image.mean(axis=[0, 1], keepdims=True)

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read points
        points = torch.load(osp.join(self.data_dir, metadata["pcd_path"]), weights_only=False)
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        # augment points
        if self.use_augmentation:
            aug_rotation = random_sample_rotation(self.aug_rotation)
            aug_transform = get_transform_from_rotation_translation(aug_rotation, None)
            points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

        # build correspondences
        if self.return_corr_indices:
            img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
            )
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["pcd_corr_indices"] = pcd_corr_indices

        # build data dict
        data_dict["intrinsics"] = intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)

        # post transform fn
        if self.transform_fn is not None:
            data_dict = self.transform_fn(data_dict)

        return data_dict
