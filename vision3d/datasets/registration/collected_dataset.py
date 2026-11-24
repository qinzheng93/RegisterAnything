from torch.utils.data import Dataset

from .kitti import OdometryKittiPairDataset
from .modelnet import ModelNetPairDataset
from .nuscenes import NuScenesPairDataset
from .threedmatch import ThreeDMatchPairDataset

DATASET_LIST = ["3DMatch", "odometryKitti", "ModelNet", "nuScenes"]


class CollectedPairDataset(Dataset):
    def __init__(
        self,
        subset: str,
        dataset_cfg_list: list[dict],
    ):
        super().__init__()

        self.subset = subset

        self.dataset_names = []
        self.dataset_list = []
        for dataset_cfg in dataset_cfg_list:
            dataset_name = dataset_cfg["name"]
            assert dataset_name in DATASET_LIST, f"Unsupport dataset: {dataset_name}"
            if dataset_name == "3DMatch":
                dataset = ThreeDMatchPairDataset(dataset_cfg["dataset_dir"], subset, **dataset_cfg["kwargs"])
            elif dataset_name == "odometryKitti":
                dataset = OdometryKittiPairDataset(dataset_cfg["dataset_dir"], subset, **dataset_cfg["kwargs"])
            elif dataset_name == "nuScenes":
                dataset = NuScenesPairDataset(dataset_cfg["dataset_dir"], subset, **dataset_cfg["kwargs"])
            elif dataset_name == "ModelNet":
                dataset = ModelNetPairDataset(dataset_cfg["dataset_dir"], subset, **dataset_cfg["kwargs"])
            else:
                raise ValueError(f"Unsupport dataset: {dataset_name}")

            self.dataset_names.append(dataset_name)
            self.dataset_list.append(dataset)

        self.total_samples = sum([len(x) for x in self.dataset_list])

    def get_dataset_index(self, index: int) -> tuple[int, int]:
        dataset_index = 0
        while index >= len(self.dataset_list[dataset_index]):
            index -= len(self.dataset_list[dataset_index])
            dataset_index += 1
        return dataset_index, index

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> dict:
        dataset_index, index = self.get_dataset_index(index)
        data_dict = self.dataset_list[dataset_index][index]
        # data_dict["dataset"] = self.dataset_names[dataset_index]
        return data_dict
