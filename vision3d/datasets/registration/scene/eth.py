from vision3d.utils.types import TransformFn

from .scene_base import ScenePairDatasetBase


class EthPairDataset(ScenePairDatasetBase):
    def __init__(
        self,
        dataset_dir: str,
        subset: str = "test",
        transform_fn: TransformFn | None = None,
        max_points: int | None = None,
        return_corr_indices: bool = False,
        matching_radius: float | None = None,
        scaling_factor: float | None = None,
        voxel_size: float = 0.2,
        min_range: float | None = None,
        max_range: float | None = None,
        max_distance: float | None = None,
        geometry_norm: bool = False,
    ):
        super().__init__(
            dataset_dir,
            subset=subset,
            transform_fn=transform_fn,
            max_points=max_points,
            return_corr_indices=return_corr_indices,
            matching_radius=matching_radius,
            scaling_factor=scaling_factor,
            voxel_size=voxel_size,
            min_range=min_range,
            max_range=max_range,
            max_distance=max_distance,
            geometry_norm=geometry_norm,
        )

    def __getitem__(self, index: int) -> dict:
        metadata: dict = self.metadata[index]

        data_dict = {}
        data_dict["dataset"] = "ETH"
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["src_frame"] = metadata["src_frame"]
        data_dict["tgt_frame"] = metadata["tgt_frame"]

        data_dict = self.generate_data(data_dict, metadata)

        return data_dict
