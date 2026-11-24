import os.path as osp

import numpy as np
import skimage.io as io
from torch.utils.data import Dataset

from vision3d.array_ops.registration_utils import get_2d3d_correspondences_mutual, get_2d3d_correspondences_radius
from vision3d.utils.io import load_pickle


class SevenScenes2D3DEasyPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        max_points: int | None = None,
        matching_method: str = "mutual_nearest",
        return_corr_indices: bool = False,
        matching_radius_3d: float = 0.015,
        matching_radius_2d: float = 12.0,
    ):
        super().__init__()

        assert subset in ["trainval", "train", "val", "test", "test_5"], f"Bad subset name: {subset}"
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))

        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_radius_3d = matching_radius_3d
        self.matching_radius_2d = matching_radius_2d
        self.matching_method = matching_method

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, index: int) -> dict:
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["seq_name"] = metadata["seq_name"]
        data_dict["image_id"] = metadata["frame_id"]
        data_dict["cloud_id"] = metadata["cloud_id"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["cloud_file"] = metadata["point_file"]

        intrinsics = metadata["intrinsic"]
        transform = metadata["point_to_image"]

        depth = io.imread(osp.join(self.data_dir, metadata["depth_file"]))
        image = io.imread(osp.join(self.data_dir, metadata["image_file"]), as_gray=True)
        image -= image.mean()

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read image
        # read points
        points = np.load(osp.join(self.data_dir, metadata["point_file"]))["points_ds"]
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        # build correspondences
        if self.return_corr_indices:
            if self.matching_method == "mutual_nearest":
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            else:
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices

        # build data dict
        data_dict["intrinsics"] = intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)

        return data_dict
