import json
import os.path as osp

import numpy as np
import skimage.transform
from torch.utils.data import Dataset

from vision3d.array_ops.ray_utils import get_camera_rays, get_world_rays
from vision3d.utils.io import read_image


class NeRFSyntheticDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        scene_name: str,
        subset: str,
        image_h: int = 800,
        image_w: int = 800,
        batched_rays: bool = False,
        batch_size: int | None = None,
        deterministic: bool = False,
        near: float = 2.0,
        far: float = 6.0,
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.scene_name = scene_name
        self.subset = subset
        self.scene_dir = osp.join(self.dataset_dir, self.scene_name)

        self.image_h = image_h
        self.image_w = image_w
        self.batched_rays = batched_rays
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.near = near
        self.far = far

        with open(osp.join(self.scene_dir, f"transforms_{self.subset}.json"), "r") as f:
            self.metadata = json.load(f)
        self.original_h = 800
        self.original_w = 800

        self.num_rays_per_image = self.image_h * self.image_w
        self.rescale = (self.image_h != self.original_h) or (self.image_w != self.original_w)
        self.scale_h = self.image_h / self.original_h
        self.scale_w = self.image_w / self.original_w

        self.focal_h = 0.5 * self.original_h / np.tan(0.5 * self.metadata["camera_angle_x"]) * self.scale_h
        self.focal_w = 0.5 * self.original_w / np.tan(0.5 * self.metadata["camera_angle_x"]) * self.scale_w
        self.center_h = 0.5 * self.original_h * self.scale_h
        self.center_w = 0.5 * self.original_w * self.scale_w

        # NeRF Synthetic dataset use OpenGL convention for the camera coordinate system,
        # where x-axis goes right, y-axis goes up and z-axis goes backward.
        self.camera_ray_directions = get_camera_rays(
            self.image_h, self.image_w, self.focal_h, self.focal_w, self.center_h, self.center_w, convention="opengl"
        )

        data_dicts = []

        frame_dicts = self.metadata["frames"]
        self.num_frames = len(frame_dicts)
        print(f"# frames: {len(frame_dicts)}")
        for frame_dict in frame_dicts:
            pose = np.asarray(frame_dict["transform_matrix"])
            filename = frame_dict["file_path"]
            image = read_image(osp.join(self.scene_dir, f"{filename}.png"))
            image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])  # RGBA -> RGB
            if self.rescale:
                image = skimage.transform.resize(image, (self.image_h, self.image_w))
            ray_centers, ray_directions = get_world_rays(self.camera_ray_directions, pose)  # (HxW, 3), (HxW, 3)
            ray_colors = image.reshape(-1, 3)  # (HxW, 3)
            data_dict = {
                "filename": filename,
                "ray_centers": ray_centers,
                "ray_directions": ray_directions,
                "ray_colors": ray_colors,
                "pose": pose,
            }
            data_dicts.append(data_dict)

        self.indices = None
        self.total_rays = self.num_frames * self.image_h * self.image_w
        self.data_dicts = None
        self.stacked_data = None

        if self.batched_rays:
            filenames = [x["filename"] for x in data_dicts]
            all_ray_centers = np.stack([x["ray_centers"] for x in data_dicts], axis=0)  # (N, HxW, 3)
            all_ray_directions = np.stack([x["ray_directions"] for x in data_dicts], axis=0)  # (N, HxW, 3)
            all_ray_colors = np.stack([x["ray_colors"] for x in data_dicts], axis=0)  # (N, HxW, 3)
            all_poses = np.stack([x["pose"] for x in data_dicts], axis=0)  # (N, 4, 4)
            self.stacked_data = {
                "filename": filenames,
                "ray_centers": all_ray_centers.astype(np.float32),
                "ray_directions": all_ray_directions.astype(np.float32),
                "ray_colors": all_ray_colors.astype(np.float32),
                "pose": all_poses.astype(np.float32),
            }
            self.indices = np.arange(self.total_rays)
        else:
            self.data_dicts = data_dicts

    def __len__(self) -> int:
        if self.batched_rays:
            return self.total_rays // self.batch_size
        else:
            return len(self.data_dicts)

    def shuffle_data(self, epoch) -> None:
        np.random.seed(epoch)
        self.indices = np.random.permutation(self.total_rays)

    def __getitem__(self, index: int) -> dict:
        # index = 0  # single-image overfitting

        new_data_dict = {"near": self.near, "far": self.far, "image_h": self.image_h, "image_w": self.image_w}

        if self.batched_rays:
            indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
            frame_indices = indices // (self.image_h * self.image_w)
            ray_indices = indices % (self.image_h * self.image_w)
            new_data_dict["filename"] = [self.stacked_data["filename"][x] for x in frame_indices.tolist()]
            new_data_dict["pose"] = self.stacked_data["pose"][frame_indices]
            new_data_dict["ray_centers"] = self.stacked_data["ray_centers"][frame_indices, ray_indices]
            new_data_dict["ray_directions"] = self.stacked_data["ray_directions"][frame_indices, ray_indices]
            new_data_dict["ray_colors"] = self.stacked_data["ray_colors"][frame_indices, ray_indices]
        else:
            data_dict = self.data_dicts[index]
            new_data_dict["filename"] = data_dict["filename"]
            new_data_dict["pose"] = data_dict["pose"].astype(np.float32)
            if self.batch_size is not None:
                # training, sample batch_size rays
                if self.deterministic:
                    np.random.seed(index)
                indices = np.random.choice(self.num_rays_per_image, self.batch_size, replace=False)
                new_data_dict["ray_centers"] = data_dict["ray_centers"][indices].astype(np.float32)
                new_data_dict["ray_directions"] = data_dict["ray_directions"][indices].astype(np.float32)
                new_data_dict["ray_colors"] = data_dict["ray_colors"][indices].astype(np.float32)
                new_data_dict["indices"] = indices
            else:
                # testing, use all rays
                new_data_dict["ray_centers"] = data_dict["ray_centers"].astype(np.float32)
                new_data_dict["ray_directions"] = data_dict["ray_directions"].astype(np.float32)
                new_data_dict["ray_colors"] = data_dict["ray_colors"].astype(np.float32)

        return new_data_dict
