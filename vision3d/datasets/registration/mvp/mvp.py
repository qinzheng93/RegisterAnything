import os.path as osp

import h5py
import numpy as np
from torch.utils.data import Dataset

from vision3d.array_ops.point_cloud_utils import random_jitter_points, random_shuffle_points
from vision3d.array_ops.se3 import apply_transform, random_sample_transform
from vision3d.utils.open3d import estimate_normals


class MVPRegistrationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        rotation_magnitude: float = 180.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: float | None = None,
        deterministic: bool = False,
        return_normals: bool = False,
        return_corr_indices: bool = False,
    ):
        super().__init__()

        assert subset in ["Train", "Test", "ExtraTest"]

        self.dataset_dir = dataset_dir
        self.subset = subset

        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.deterministic = deterministic
        self.return_normals = return_normals
        self.return_corr_indices = return_corr_indices

        self.data_list = self.load_data()

    def load_data(self) -> list[dict]:
        filename = osp.join(self.dataset_dir, "MVP_{}_RG.h5".format(self.subset))
        database = h5py.File(filename, "r")
        all_src_points = database["src"][:]
        all_tgt_points = database["tgt"][:]
        all_labels = database["cat_labels"][:]
        all_raw_points = database["complete"][:]
        match_id = database["match_id"]
        if "transforms" in database:
            transforms = database["transforms"][:]
        else:
            transforms = None
        num_pairs = all_tgt_points.shape[0]
        data_list = []
        for i in range(num_pairs):
            data_dict = {
                "raw_points": all_raw_points[i].copy(),
                "src_points": all_src_points[i].copy(),
                "tgt_points": all_tgt_points[i].copy(),
                "label": all_labels[i],
            }
            if self.return_corr_indices:
                data_dict["corr_indices"] = match_id[str(i)][:].copy()
            if transforms is not None:
                data_dict["transform"] = transforms[i].copy()
            data_list.append(data_dict)
        return data_list

    def __getitem__(self, index: int) -> dict:
        data_dict: dict = self.data_list[index]
        src_points = data_dict["src_points"].copy()
        tgt_points = data_dict["tgt_points"].copy()
        raw_points = data_dict["raw_points"].copy()
        label = data_dict["label"]

        if "transforms" not in data_dict:
            # random transform to source point cloud
            src_transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            src_points = apply_transform(src_points, src_transform)

            # random transform to target point cloud
            tgt_transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            tgt_points = apply_transform(tgt_points, tgt_transform)

            transform = np.matmul(tgt_transform, np.linalg.inv(src_transform))
        else:
            transform = data_dict["transform"].copy()

        # random jitter
        if self.noise_magnitude is not None:
            src_points = random_jitter_points(src_points, sigma=0.01, scale=self.noise_magnitude)
            tgt_points = random_jitter_points(tgt_points, sigma=0.01, scale=self.noise_magnitude)

        # random shuffle
        src_points = random_shuffle_points(src_points)
        tgt_points = random_shuffle_points(tgt_points)

        new_data_dict = {
            "raw_points": raw_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "tgt_points": tgt_points.astype(np.float32),
            "transform": transform.astype(np.float32),
            "label": int(label),
            "index": int(index),
        }

        if self.return_normals:
            src_normals = estimate_normals(src_points)
            new_data_dict["src_normals"] = src_normals.astype(np.float32)
            tgt_normals = estimate_normals(tgt_points)
            new_data_dict["tgt_normals"] = tgt_normals.astype(np.float32)

        return new_data_dict

    def __len__(self) -> int:
        return len(self.data_list)
