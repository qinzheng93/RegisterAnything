import os.path as osp

import cv2
import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset

from vision3d.array_ops.depth_image import back_project
from vision3d.array_ops.se3 import compose_transforms, inverse_transform
from vision3d.utils.io import load_pickle, read_depth_image
from vision3d.utils.open3d import voxel_down_sample


class ThreeDMatchPairURRCropDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        voxel_size: float = 0.025,
        depth_limit: float | None = 6.0,
        scaling_factor: float = 1000.0,
        image_h: int = 128,
        image_w: int = 128,
        max_points: int | None = None,
        overfitting: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"], f"Unsupported data subset: {subset}"
        assert image_h == image_w

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "frames")
        self.metadata_dir = osp.join(self.dataset_dir, "urnr", "metadata")
        self.subset = subset
        self.voxel_size = voxel_size
        self.depth_limit = depth_limit
        self.scaling_factor = scaling_factor
        self.image_h = image_h
        self.image_w = image_w
        self.max_points = max_points
        self.overfitting = overfitting

        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))

    def __len__(self) -> int:
        return len(self.metadata_list)

    def read_frame(self, scene_name: str, seq_name: str, frame_name: str) -> tuple[ndarray, ndarray, ndarray]:
        # read data
        seq_dir = osp.join(self.data_dir, scene_name, seq_name)
        if scene_name.startswith("7-scenes"):
            color_img = cv2.imread(osp.join(seq_dir, f"{frame_name}.calib.png"))  # (H, W, 3)
        else:
            color_img = cv2.imread(osp.join(seq_dir, f"{frame_name}.color.png"))  # (H, W, 3)
        depth_img = read_depth_image(osp.join(seq_dir, f"{frame_name}.depth.png"))  # (H, W)
        pose = np.loadtxt(osp.join(seq_dir, f"{frame_name}.pose.txt"))

        return color_img, depth_img, pose

    def read_points(self, scene_name: str, seq_name: str, frame_name: str) -> ndarray:
        # read preprocessed point cloud data
        points = np.load(osp.join(self.data_dir, scene_name, seq_name, f"{frame_name}.cloud.npy"))

        # random sample
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]

        return points

    def get_points(self, depth_img: ndarray, intrinsics: ndarray) -> ndarray:
        # convert to point cloud
        points = back_project(depth_img, intrinsics, scaling_factor=self.scaling_factor, depth_limit=self.depth_limit)

        # voxelize
        if self.voxel_size is not None:
            points = voxel_down_sample(points, self.voxel_size)

        # random sample
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]

        return points

    def align_intrinsics(
        self, color_img: ndarray, depth_img: ndarray, color_intrinsics: ndarray, depth_intrinsics: ndarray
    ) -> ndarray:
        scale_h = depth_intrinsics[1, 1] / color_intrinsics[1, 1]
        scale_w = depth_intrinsics[0, 0] / color_intrinsics[0, 0]
        new_image_h = int(scale_h * color_img.shape[0])
        new_image_w = int(scale_w * color_img.shape[1])
        color_img = cv2.resize(color_img, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
        image_h = depth_img.shape[0]
        image_w = depth_img.shape[1]
        start_index_h = new_image_h // 2 - image_h // 2
        end_index_h = start_index_h + image_h
        start_index_w = new_image_w // 2 - image_w // 2
        end_index_w = start_index_w + image_w
        color_img = color_img[start_index_h:end_index_h, start_index_w:end_index_w]
        return color_img

    def img_transform(
        self,
        src_img: ndarray,
        tgt_img: ndarray,
        intrinsics: ndarray,
        normalize: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> tuple[ndarray, ndarray, ndarray]:
        # 1. resize
        image_h = self.image_h
        scale = image_h / src_img.shape[0]
        image_w = int(scale * src_img.shape[1])

        new_src_img = cv2.resize(src_img, (image_w, image_h), interpolation=interpolation)
        new_tgt_img = cv2.resize(tgt_img, (image_w, image_h), interpolation=interpolation)

        # 2. center crop
        w_start_index = image_w // 2 - image_h // 2
        w_end_index = w_start_index + image_h
        new_src_img = new_src_img[:, w_start_index:w_end_index]
        new_tgt_img = new_tgt_img[:, w_start_index:w_end_index]

        # 3. update intrinsics
        new_intrinsics = np.eye(3)
        new_intrinsics[0, 0] = scale * intrinsics[0, 0]
        new_intrinsics[0, 2] = 0.5 * image_h
        new_intrinsics[1, 1] = scale * intrinsics[1, 1]
        new_intrinsics[1, 2] = 0.5 * image_h

        # 4. normalize color image
        if normalize:
            new_src_img = 2.0 * (new_src_img / 255.0 - 0.5)
            new_tgt_img = 2.0 * (new_tgt_img / 255.0 - 0.5)

        return new_src_img, new_tgt_img, new_intrinsics

    def pcd_transform(self, points: ndarray) -> ndarray:
        # TODO: point transform
        pass

    def __getitem__(self, index: int) -> dict:
        if self.overfitting:
            index = 100000

        metadata_dict: dict = self.metadata_list[index]

        scene_name = metadata_dict["scene_name"]
        seq_name = metadata_dict["seq_name"]
        src_frame = metadata_dict["src_frame"]
        tgt_frame = metadata_dict["tgt_frame"]

        color_intrinsics = metadata_dict["depth_intrinsics"]
        depth_intrinsics = metadata_dict["depth_intrinsics"]

        src_color_img, src_depth_img, src_pose = self.read_frame(scene_name, seq_name, src_frame)
        tgt_color_img, tgt_depth_img, tgt_pose = self.read_frame(scene_name, seq_name, tgt_frame)

        src_points = self.get_points(src_depth_img, depth_intrinsics)
        tgt_points = self.get_points(tgt_depth_img, depth_intrinsics)
        # src_points = self.read_points(scene_name, seq_name, src_frame)
        # tgt_points = self.read_points(scene_name, seq_name, tgt_frame)

        tgt_inv_pose = inverse_transform(tgt_pose)
        transform = compose_transforms(src_pose, tgt_inv_pose)

        # if scene_name.startswith("7-scenes"):
        #     # color and depth intrinsics are not aligned, fix this
        #     src_color_img = self.align_intrinsics(src_color_img, src_depth_img, color_intrinsics, depth_intrinsics)
        #     tgt_color_img = self.align_intrinsics(tgt_color_img, tgt_depth_img, color_intrinsics, depth_intrinsics)

        src_color_img, tgt_color_img, color_intrinsics = self.img_transform(
            src_color_img, tgt_color_img, color_intrinsics, normalize=True, interpolation=cv2.INTER_LINEAR
        )

        src_depth_img, tgt_depth_img, depth_intrinsics = self.img_transform(
            src_depth_img, tgt_depth_img, depth_intrinsics, normalize=False, interpolation=cv2.INTER_NEAREST
        )

        # src_color_img, src_depth_img = self.img_transform(src_color_img, src_depth_img)
        # tgt_color_img, tgt_depth_img = self.img_transform(tgt_color_img, tgt_depth_img)
        # color_intrinsics, depth_intrinsics = self.intrinsic_transform(color_intrinsics, depth_intrinsics)
        # src_points = self.pcd_transform(src_points)
        # tgt_points = self.pcd_transform(tgt_points)

        output_dict = {
            "scene_name": scene_name,
            "seq_name": seq_name,
            "src_frame": src_frame,
            "tgt_frame": tgt_frame,
            "color_intrinsics": color_intrinsics.astype(np.float32),
            "depth_intrinsics": depth_intrinsics.astype(np.float32),
            "src_color_img": src_color_img.astype(np.float32),
            "tgt_color_img": tgt_color_img.astype(np.float32),
            "src_depth_img": src_depth_img.astype(np.float32),
            "tgt_depth_img": tgt_depth_img.astype(np.float32),
            "src_pose": src_pose.astype(np.float32),
            "tgt_pose": tgt_pose.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "tgt_points": tgt_points.astype(np.float32),
            "transform": transform.astype(np.float32),
        }

        return output_dict
