import os.path as osp

import cv2
import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.depth_image import back_project
from vision3d.array_ops.registration_utils import get_2d3d_correspondences_mutual, get_2d3d_correspondences_radius
from vision3d.array_ops.se3 import (
    apply_transform,
    compose_transforms,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
)
from vision3d.utils.io import load_pickle, read_depth_image, read_image
from vision3d.utils.open3d import voxel_down_sample


class ScanNet2D3DPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        max_points: int | None = None,
        return_corr_indices: bool = False,
        matching_method: str = "mutual_nearest",
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.0375,
        image_h: int = 640,
        image_w: int = 480,
        voxel_size: float = 0.025,
        scaling_factor: float = 1000.0,
        depth_limit: float = 6.0,
        scene_name: str | None = None,
        overlap_threshold: float | None = 0.5,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        scale_augmentation: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"], f"Bad subset name: {subset}."
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))

        if scene_name is not None:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_name]

        if overlap_threshold is not None:
            self.metadata_list = [
                x
                for x in self.metadata_list
                if x["src_overlap"] >= overlap_threshold and x["tgt_overlap"] >= overlap_threshold
            ]

        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_method = matching_method
        self.matching_radius_2d = matching_radius_2d
        self.matching_radius_3d = matching_radius_3d
        self.overlap_threshold = overlap_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.scale_augmentation = scale_augmentation
        self.image_h = image_h
        self.image_w = image_w
        self.voxel_size = voxel_size
        self.scaling_factor = scaling_factor
        self.depth_limit = depth_limit

    def __len__(self) -> int:
        return len(self.metadata_list)

    def get_points(self, depth_img: ndarray, intrinsics: ndarray) -> ndarray:
        # backproject to point cloud
        points = back_project(depth_img, intrinsics, scaling_factor=self.scaling_factor, depth_limit=self.depth_limit)

        # voxelize
        if self.voxel_size is not None:
            points = voxel_down_sample(points, self.voxel_size)

        return points

    def image_transform(self, img: ndarray, intrinsics: ndarray) -> tuple[ndarray, ndarray]:
        # 1. resize
        scale_h = self.image_h / img.shape[0]
        scale_w = self.image_w / img.shape[1]

        new_img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

        new_intrinsics = np.eye(3)
        new_intrinsics[0, 0] = scale_w * intrinsics[0, 0]
        new_intrinsics[0, 2] = scale_w * intrinsics[0, 2]
        new_intrinsics[1, 1] = scale_h * intrinsics[1, 1]
        new_intrinsics[1, 2] = scale_h * intrinsics[1, 2]

        # 2. normalize color image
        new_img = 2.0 * (new_img / 255.0 - 0.5)

        return new_img, new_intrinsics

    def depth_transform(self, img: ndarray, intrinsics: ndarray) -> tuple[ndarray, ndarray]:
        # 1. resize
        scale_h = self.image_h / img.shape[0]
        scale_w = self.image_w / img.shape[1]

        new_img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)

        new_intrinsics = np.eye(3)
        new_intrinsics[0, 0] = scale_w * intrinsics[0, 0]
        new_intrinsics[0, 2] = scale_w * intrinsics[0, 2]
        new_intrinsics[1, 1] = scale_h * intrinsics[1, 1]
        new_intrinsics[1, 2] = scale_h * intrinsics[1, 2]

        return new_img, new_intrinsics

    def __getitem__(self, index: int) -> dict:
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["point_file"] = metadata["src_point_path"]
        data_dict["image_file"] = metadata["tgt_image_path"]
        data_dict["depth_file"] = metadata["tgt_depth_path"]
        data_dict["src_overlap"] = metadata["src_overlap"]
        data_dict["tgt_overlap"] = metadata["tgt_overlap"]
        data_dict["point_id"] = metadata["src_frame"]
        data_dict["image_id"] = metadata["tgt_frame"]
        data_dict["image_h"] = self.image_h
        data_dict["image_w"] = self.image_w

        image_intrinsics = metadata["color_intrinsics"]
        depth_intrinsics = metadata["depth_intrinsics"]
        transform = metadata["transform"]

        # read image
        image = read_image(osp.join(self.data_dir, metadata["tgt_image_path"]), as_gray=True)
        depth = read_depth_image(osp.join(self.data_dir, metadata["tgt_depth_path"])).astype(np.float32)

        # read points
        src_depth = read_depth_image(osp.join(self.data_dir, metadata["src_depth_path"])).astype(np.float32)
        points = self.get_points(src_depth, depth_intrinsics)
        # points = np.load(osp.join(self.data_dir, metadata["src_point_path"]))
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        # resize image
        image, image_intrinsics = self.image_transform(image, image_intrinsics)
        depth, depth_intrinsics = self.depth_transform(depth, depth_intrinsics)

        if self.use_augmentation:
            # augment point cloud
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

        # build correspondences
        if self.return_corr_indices:
            if self.matching_method == "mutual_nearest":
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                    depth, points, depth_intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            else:
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                    depth, points, depth_intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            img_corr_indices = img_corr_pixels[:, 0] * self.image_w + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices

        # build data dict
        data_dict["image_intrinsics"] = image_intrinsics.astype(np.float32)
        data_dict["depth_intrinsics"] = depth_intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)

        return data_dict
