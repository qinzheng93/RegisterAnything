import os.path as osp

import numpy as np
import skimage.transform
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.ray_utils import batch_get_world_rays, get_camera_rays, get_world_rays
from vision3d.utils.io import load_pickle, read_image


class SevenScenesNeRFDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        scene_name: str,
        subset: str,
        image_h: int = 480,
        image_w: int = 640,
        batched_rays: bool = False,
        batch_size: int | None = None,
        deterministic: bool = False,
        near: float = 0.1,
        far: float = 5.0,
        frame_stride: int = 1,
        pose_method: str = "sfm",
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata", pose_method)
        self.scene_name = scene_name
        self.subset = subset
        self.pose_method = pose_method

        self.image_h = image_h
        self.image_w = image_w
        self.batched_rays = batched_rays
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.near = near
        self.far = far

        self.metadata = load_pickle(osp.join(self.metadata_dir, f"{self.scene_name}.{self.subset}.pkl"))
        self.intrinsics = self.metadata["intrinsics"]
        self.original_h = self.metadata["original_h"]
        self.original_w = self.metadata["original_w"]

        self.num_rays_per_image = self.image_h * self.image_w
        self.rescale = (self.image_h != self.original_h) or (self.image_w != self.original_w)
        self.scale_h = self.image_h / self.original_h
        self.scale_w = self.image_w / self.original_w

        self.focal_h = self.intrinsics[1, 1] * self.scale_h
        self.focal_w = self.intrinsics[0, 0] * self.scale_w
        self.center_h = self.intrinsics[1, 2] * self.scale_h
        self.center_w = self.intrinsics[0, 2] * self.scale_w

        self.camera_ray_directions = get_camera_rays(
            self.image_h, self.image_w, self.focal_h, self.focal_w, self.center_h, self.center_w
        ).reshape(-1, 3)

        data_dicts = self.metadata["frames"]
        if frame_stride > 1:
            data_dicts = data_dicts[::frame_stride]
        self.data_dicts = data_dicts
        self.num_frames = len(data_dicts)
        self.total_rays = self.num_frames * self.image_h * self.image_w

        print(f"# frames: {self.num_frames}")

        # for sample_all mode
        self.indices = None
        self.stacked_data = None

        if self.batched_rays:
            all_filename = []
            all_pose = []
            all_ray_colors = []
            for data_dict in data_dicts:
                pose = data_dict["pose"]
                filename = data_dict["image"]
                ray_colors = self.read_image(filename).reshape(-1, 3)
                all_filename.append(filename)
                all_pose.append(pose)
                all_ray_colors.append(ray_colors)
            self.stacked_data = {
                "filename": all_filename,
                "pose": np.stack(all_pose, axis=0),  # (N, 4, 4)
                "ray_colors": np.stack(all_ray_colors, axis=0),  # (N, HxW, 3)
            }

    def read_image(self, filename: str) -> ndarray:
        image = read_image(osp.join(self.data_dir, self.scene_name, filename))
        if self.rescale:
            image = skimage.transform.resize(image, (self.image_h, self.image_w))
        return image  # type: ignore

    def __len__(self) -> int:
        if self.batched_rays:
            assert self.batch_size is not None
            return self.total_rays // self.batch_size
        return len(self.data_dicts)

    def shuffle_data(self, epoch: int) -> None:
        np.random.seed(epoch)
        self.indices = np.random.permutation(self.total_rays)

    def __getitem__(self, index: int) -> dict:
        # index = 0  # single-image overfitting

        new_data_dict = {
            "near": self.near,
            "far": self.far,
            "image_h": self.image_h,
            "image_w": self.image_w,
            "intrinsics": self.intrinsics.astype(np.float32),
        }

        if self.batched_rays:
            indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
            frame_indices = indices // (self.image_h * self.image_w)
            ray_indices = indices % (self.image_h * self.image_w)

            pose = self.stacked_data["pose"][frame_indices]
            camera_ray_directions = self.camera_ray_directions[ray_indices]
            ray_centers, ray_directions = batch_get_world_rays(camera_ray_directions, pose)
            ray_colors = self.stacked_data["ray_colors"][frame_indices, ray_indices]

            new_data_dict["filename"] = [self.stacked_data["filename"][x] for x in frame_indices]
            new_data_dict["pose"] = pose.astype(np.float32)
            new_data_dict["ray_centers"] = ray_centers.astype(np.float32)
            new_data_dict["ray_directions"] = ray_directions.astype(np.float32)
            new_data_dict["ray_colors"] = ray_colors.astype(np.float32)
        else:
            data_dict = self.data_dicts[index]
            filename = data_dict["image"]
            pose = data_dict["pose"]

            ray_centers, ray_directions = get_world_rays(self.camera_ray_directions, pose)
            ray_colors = self.read_image(filename).reshape(-1, 3)

            if self.batch_size is not None:
                # training, sample batch_size rays
                if self.deterministic:
                    np.random.seed(index)
                indices = np.random.choice(self.num_rays_per_image, self.batch_size, replace=False)
                ray_centers = ray_centers[indices]
                ray_directions = ray_directions[indices]
                ray_colors = ray_colors[indices]
                new_data_dict["ray_indices"] = indices

            new_data_dict["filename"] = filename
            new_data_dict["ray_centers"] = ray_centers.astype(np.float32)
            new_data_dict["ray_directions"] = ray_directions.astype(np.float32)
            new_data_dict["ray_colors"] = ray_colors.astype(np.float32)
            new_data_dict["pose"] = pose.astype(np.float32)

        return new_data_dict
