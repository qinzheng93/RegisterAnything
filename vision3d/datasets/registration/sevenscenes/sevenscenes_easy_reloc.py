import numpy as np
import skimage.io as io
from torch.utils.data import Dataset

from vision3d.array_ops.se3 import compose_transforms, inverse_transform
from vision3d.utils.io import readlines
from vision3d.utils.open3d import voxel_down_sample


class SevenScenes2D3DEasyRelocPairDataset(Dataset):
    def __init__(
        self,
        scene_name: str,
        subset: str,
        max_points: int | None = None,
        voxel_size: float = 0.025,
        start_index: int = 0,
    ):
        super().__init__()

        self.intrinsics = np.asarray([[585.0, 0.0, 320], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]])

        self.subset = subset
        self.scene_name = scene_name
        self.voxel_size = voxel_size

        self.start_index = start_index

        self.pose_dir = f"/data/7Scenes/7scenes-frames/data/{self.scene_name}"
        self.data_dir = f"/data/7Scenes/7scenes-2d3d/data/{self.scene_name}"

        self.max_points = max_points

        lines = readlines(f"/data/7Scenes/7scenes-retrieval/{self.subset}/{self.scene_name}_top10.txt")
        lines = lines[self.start_index :]
        metadata_list = []
        for line in lines:
            split_line = line.split()
            src_seq_name, src_frame_id = split_line[0].split("/")
            tgt_seq_name, tgt_frame_id = split_line[1].split("/")
            src_frame_id = src_frame_id.split(".")[0]
            tgt_frame_id = tgt_frame_id.split(".")[0]
            metadata = {
                "src_seq_name": src_seq_name,
                "src_frame_id": src_frame_id,
                "tgt_seq_name": tgt_seq_name,
                "tgt_frame_id": tgt_frame_id,
            }
            metadata_list.append(metadata)
        self.metadata_list = metadata_list

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, index: int) -> dict:
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        q_seq_name = metadata["src_seq_name"]
        q_frame_id = metadata["src_frame_id"]
        s_seq_name = metadata["tgt_seq_name"]
        s_frame_id = metadata["tgt_frame_id"]

        # read query image
        depth = io.imread(f"{self.data_dir}/{q_seq_name}/depth/{q_frame_id}-resized.depth.png")
        image = io.imread(f"{self.data_dir}/{q_seq_name}/rgb/{q_frame_id}-resized.color.png", as_gray=True)
        image -= image.mean()

        q_pose = np.loadtxt(f"{self.pose_dir}/{q_seq_name}/poses/{q_frame_id}.pose.txt")

        # read support point cloud
        pcd_id = int(s_frame_id.split("-")[-1]) // 5
        s_pose_id = pcd_id * 5

        points = np.load(f"{self.data_dir}/{s_seq_name}/points_ds/cloud_bin_{pcd_id}.npz")["points_ds"]
        if self.voxel_size is not None:
            points = voxel_down_sample(points, self.voxel_size)
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        s_pose = np.loadtxt(f"{self.pose_dir}/{s_seq_name}/poses/frame-{s_pose_id:06d}.pose.txt")

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # compute transform
        transform = compose_transforms(s_pose, inverse_transform(q_pose))

        # build data dict
        data_dict["scene_name"] = self.scene_name
        data_dict["q_seq_name"] = q_seq_name
        data_dict["q_frame_id"] = q_frame_id
        data_dict["s_seq_name"] = s_seq_name
        data_dict["s_frame_id"] = s_frame_id
        data_dict["intrinsics"] = self.intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)
        data_dict["q_pose"] = q_pose.astype(np.float32)
        data_dict["s_pose"] = s_pose.astype(np.float32)

        return data_dict
