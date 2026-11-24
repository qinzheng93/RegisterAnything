import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset

from vision3d.utils.types import TransformFn


class FlyingThingsPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: TransformFn | None = None,
        positive_threshold: float = 0.05,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.002,
    ):
        super().__init__()

        assert subset in ["train", "val"]

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.data_dir = osp.join(self.dataset_dir, self.subset)

        self.transform_fn = transform_fn
        self.positive_threshold = positive_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise

        self.filenames = sorted(os.listdir(self.data_dir))
        self.filenames = [filename for filename in self.filenames if filename.endswith(".npz")]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        new_data_dict = {}

        filename = self.filenames[index]
        data_dict = np.load(osp.join(self.data_dir, filename))

        new_data_dict["filename"] = osp.join(self.subset, filename)

        src_points = data_dict["src_points"]
        tgt_points = data_dict["tgt_points"]
        scene_flows = data_dict["scene_flows"]
        estimated_scene_flows = data_dict["estimated_scene_flows"]
        warped_src_points = src_points + scene_flows
        src_corr_points = src_points
        tgt_corr_points = src_corr_points + estimated_scene_flows

        if self.use_augmentation:
            # aug_transform = random_sample_small_transform()
            # if random.random() > 0.5:
            #     tgt_center = tgt_points.mean(axis=0)
            #     subtract_center = get_transform_from_rotation_translation(None, -tgt_center)
            #     add_center = get_transform_from_rotation_translation(None, tgt_center)
            #     aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            #     tgt_points = apply_transform(tgt_points, aug_transform)
            #     tgt_corr_points = apply_transform(tgt_corr_points, aug_transform)
            # else:
            #     src_center = src_points.mean(axis=0)
            #     subtract_center = get_transform_from_rotation_translation(None, -src_center)
            #     add_center = get_transform_from_rotation_translation(None, src_center)
            #     aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            #     src_points = apply_transform(src_points, aug_transform)
            #     warped_src_points = apply_transform(warped_src_points, aug_transform)
            #     src_corr_points = apply_transform(src_corr_points, aug_transform)

            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
            tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise
            src_corr_points += (np.random.rand(src_corr_points.shape[0], 3) - 0.5) * self.aug_noise
            tgt_corr_points += (np.random.rand(tgt_corr_points.shape[0], 3) - 0.5) * self.aug_noise

        scene_flows = warped_src_points - src_points
        corr_scene_flows = tgt_corr_points - src_corr_points

        absolute_error = np.linalg.norm(corr_scene_flows - scene_flows, axis=1)
        relative_error = absolute_error / (np.linalg.norm(scene_flows, axis=1) + 1e-6)
        corr_labels = (absolute_error < 0.05) | (relative_error < 0.05)

        new_data_dict["src_points"] = src_points.astype(np.float32)
        new_data_dict["tgt_points"] = tgt_points.astype(np.float32)
        new_data_dict["src_feats"] = np.ones(shape=(src_points.shape[0], 1), dtype=np.float32)
        new_data_dict["tgt_feats"] = np.ones(shape=(tgt_points.shape[0], 1), dtype=np.float32)
        new_data_dict["scene_flows"] = scene_flows.astype(np.float32)
        new_data_dict["src_corr_points"] = src_corr_points.astype(np.float32)
        new_data_dict["tgt_corr_points"] = tgt_corr_points.astype(np.float32)
        new_data_dict["corr_scene_flows"] = corr_scene_flows.astype(np.float32)
        new_data_dict["corr_labels"] = corr_labels.astype(np.int64)
        new_data_dict["transform"] = np.eye(4).astype(np.float32)

        if self.transform_fn is not None:
            new_data_dict = self.transform_fn(new_data_dict)

        return new_data_dict
