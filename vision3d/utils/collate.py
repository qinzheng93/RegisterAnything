from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from vision3d.array_ops.graph_pyramid import (
    build_grid_pyramid_with_knn_graph_pack_mode,
    build_grid_pyramid_with_radius_graph_pack_mode,
    build_uniformized_grid_pyramid_with_radius_graph_pack_mode,
)
from vision3d.array_ops.se3 import scale_transform

from .tensor import array_to_tensor


def collate_dict(data_dicts: list[dict]) -> dict:
    """Collate a batch of dict.

    The collated dict contains all keys from the batch, with each key mapped to a list of data. If a certain key is
    missing in one dict, `None` is used for padding so that all lists have the same length (the batch size).

    Args:
        data_dicts (list[dict]): A batch of data dicts.

    Returns:
        A dict with all data collated.
    """
    keys = set(chain(*[list(data_dict.keys()) for data_dict in data_dicts]))
    collated_dict = {key: [data_dict.get(key) for data_dict in data_dicts] for key in keys}
    return collated_dict


def point_cloud_collate_fn_pack_mode(data_dicts: list[dict], unpack_single: bool = True) -> dict:
    batch_size = len(data_dicts)

    # merge data with the same key from different samples into a list
    collated_dict = collate_dict(data_dicts)

    if batch_size == 1 and unpack_single:
        # unpack list if batch_size is 1
        collated_dict = {key: value[0] for key, value in collated_dict.items()}
        collated_dict["lengths"] = np.asarray([collated_dict["points"].shape[0]])
    else:
        # pack points, feats, labels and generate lengths
        points_list = collated_dict.pop("points")
        collated_dict["points"] = np.concatenate(points_list, axis=0)
        collated_dict["lengths"] = np.asarray([points.shape[0] for points in points_list])
        if "feats" in collated_dict:
            collated_dict["feats"] = np.concatenate(collated_dict.pop("feats"), axis=0)
        if "labels" in collated_dict:
            collated_dict["labels"] = np.concatenate(collated_dict.pop("labels"), axis=0)

    collated_dict["batch_size"] = batch_size

    return collated_dict


def registration_collate_fn_pack_mode(data_dicts: list[dict], unpack_single: bool = True) -> dict:
    batch_size = len(data_dicts)

    # merge data with the same key from different samples into a list
    collated_dict = collate_dict(data_dicts)

    if batch_size == 1 and unpack_single:
        # unpack list if batch_size is 1
        collated_dict = {key: value[0] for key, value in collated_dict.items()}
        collated_dict["src_lengths"] = np.asarray([collated_dict["src_points"].shape[0]])
        collated_dict["tgt_lengths"] = np.asarray([collated_dict["tgt_points"].shape[0]])
    else:
        # handle special keys: [src_feats, tgt_feats], [src_points, tgt_points], [src_lengths, tgt_lengths]
        src_points_list = collated_dict.pop("src_points")
        tgt_points_list = collated_dict.pop("tgt_points")
        collated_dict["src_points"] = np.concatenate(src_points_list, axis=0)
        collated_dict["tgt_points"] = np.concatenate(tgt_points_list, axis=0)
        collated_dict["src_lengths"] = np.asarray([src_points.shape[0] for src_points in src_points_list])
        collated_dict["tgt_lengths"] = np.asarray([tgt_points.shape[0] for tgt_points in tgt_points_list])
        if "src_feats" in collated_dict:
            collated_dict["src_feats"] = np.concatenate(collated_dict.pop("src_feats"), axis=0)
        if "tgt_feats" in collated_dict:
            collated_dict["tgt_feats"] = np.concatenate(collated_dict.pop("tgt_feats"), axis=0)

    collated_dict["batch_size"] = batch_size

    return collated_dict


class CollateFnBase(ABC):
    @abstractmethod
    def __call__(self, data_dicts: list[dict]) -> dict:
        pass


class CollateFnPackMode(CollateFnBase):
    def __init__(self, unpack_single: bool = True):
        self.unpack_single = unpack_single


class CollateDictFnPackMode(CollateFnPackMode):
    def __call__(self, data_dicts: list[dict]) -> dict:
        batch_size = len(data_dicts)

        collated_dict = collate_dict(data_dicts)

        if batch_size == 1 and self.unpack_single:
            # unpack list if batch_size is 1
            collated_dict = {key: value[0] for key, value in collated_dict.items()}

        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


def get_max_neighbors(radius: float) -> int:
    max_neighbors = int(np.ceil(4 / 3 * np.pi * (radius + 1) ** 3))
    return max_neighbors


class CollateFnPackModeWithGraphPyramid(CollateFnBase):
    def __init__(
        self,
        num_stages: int,
        voxel_size: float,
        search_radius: float,
        neighbor_limits: list[int] | int | None = None,
        unpack_single: bool = True,
    ):
        self.num_stages = num_stages
        self.voxel_size = voxel_size
        self.search_radius = search_radius
        self.unpack_single = unpack_single

        if neighbor_limits is None:
            neighbor_limits = get_max_neighbors(search_radius / voxel_size)
        if isinstance(neighbor_limits, int):
            neighbor_limits = [neighbor_limits] * num_stages
        self.neighbor_limits: list[int] = neighbor_limits


class CollateFnPackModeWithUniformizedGraphPyramid(CollateFnPackModeWithGraphPyramid):
    def __init__(
        self,
        num_stages: int,
        voxel_size: float,
        search_radius: float,
        num_neighbors: int,
        distance_limit: float,
        neighbor_limits: list[int] | int | None = None,
        unpack_single: bool = True,
    ):
        super().__init__(
            num_stages, voxel_size, search_radius, neighbor_limits=neighbor_limits, unpack_single=unpack_single
        )
        self.num_neighbors = num_neighbors
        self.distance_limit = distance_limit


class CollateFnPackModeWithKnnGraphPyramid(CollateFnBase):
    def __init__(
        self,
        num_stages: int,
        voxel_size: float,
        num_neighbors: int,
        unpack_single: bool = True,
    ):
        self.num_stages = num_stages
        self.voxel_size = voxel_size
        self.num_neighbors = num_neighbors
        self.unpack_single = unpack_single


class PointCloudCollateFnPackMode(CollateFnPackMode):
    """Simple collate function for single point cloud in pack mode.

    Note:
        1. The data of keys "points", "feats" and "labels" are packed into large tensors by stacking along axis 0. The
            names of the packed tensors are the same.
        2. A new tensor named "lengths" contains the length of each sample in the batch.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        collated_dict = point_cloud_collate_fn_pack_mode(data_dicts, unpack_single=self.unpack_single)
        collated_dict = array_to_tensor(collated_dict)
        return collated_dict


class PointCloudCollateFnPackModeWithGraphPyramid(CollateFnPackModeWithGraphPyramid):
    """Collate function for single point cloud in pack mode with graph pyramid.

    Note:
        1. The data of keys "points", "feats" and "labels" are packed into large tensors by stacking along axis 0.
        2. The names of packed feature tensor and the label tensor are "feats" and "labels" respectively.
        3. This collation function build a graph pyramid with grid subsampling and radius neighbor searching. The keys
            "points", "lengths", "neighbors", "subsampling" and "upsampling" are five `list` holding the information of
            the graph pyramid.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        collated_dict = point_cloud_collate_fn_pack_mode(data_dicts, unpack_single=self.unpack_single)

        # build graph pyramid
        points = collated_dict.pop("points")
        lengths = collated_dict.pop("lengths")
        graph_pyramid_dict = build_grid_pyramid_with_radius_graph_pack_mode(
            points, lengths, self.num_stages, self.voxel_size, self.search_radius, self.neighbor_limits
        )
        collated_dict.update(graph_pyramid_dict)

        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class S3DISTestingCollateFnPackModeWithGraphPyramid(CollateFnBase):
    def __init__(
        self,
        num_stages: int,
        voxel_size: float,
        search_radius: float,
        neighbor_limits: list[int],
        batch_size: int,
        unpack_single: bool = True,
    ):
        self.batch_size = batch_size
        self.collate_fn = PointCloudCollateFnPackModeWithGraphPyramid(
            num_stages, voxel_size, search_radius, neighbor_limits=neighbor_limits, unpack_single=unpack_single
        )

    def __call__(self, data_dicts: list[dict]) -> dict:
        assert len(data_dicts) == 1, "Only support batch size of 1 at this time."

        data_dict = data_dicts[0]

        batch_list = data_dict["batch_list"]
        num_batches = len(batch_list)
        new_batch_list = []
        for i in range(0, num_batches, self.batch_size):
            end_index = min(i + self.batch_size, num_batches)
            cur_data_batch = batch_list[i:end_index]
            cur_data_batch = self.collate_fn(cur_data_batch)
            if isinstance(cur_data_batch["indices"], list):
                cur_data_batch["indices"] = torch.cat(cur_data_batch["indices"], dim=0)
            new_batch_list.append(cur_data_batch)

        return {
            "labels": torch.from_numpy(data_dict["labels"]),
            "batch_list": new_batch_list,
        }


class RegistrationCollateFnPackMode(CollateFnPackMode):
    """Simple collate function for registration in pack mode.

    Note:
        1. The data of keys "src_points", "tgt_points", "src_feats" and "tgt_feats" are packed into large tensors by
            stacking along axis 0. The names of the packed tensors are the same.
        2. Unlike `GraphPyramidRegistrationCollateFnPackMode`, the source point clouds and the target point clouds are
            NOT fused in this collate function.
        3. A new tensor named "lengths" contains the length of each sample in the batch.
        4. The correspondence indices are within each point cloud without accumulation over the batch.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        collated_dict = registration_collate_fn_pack_mode(data_dicts, unpack_single=self.unpack_single)
        collated_dict = array_to_tensor(collated_dict)
        return collated_dict


class RegistrationCollateFnPackModeWithGraphPyramid(CollateFnPackModeWithGraphPyramid):
    """Collate function for registration in pack mode with graph pyramid.

    Note:
        1. The data of keys "src_points", "tgt_points", "src_feats" and "tgt_feats" are packed into large tensors by
            stacking along axis 0. The source point clouds and the target point clouds are then merged into a fused
            point cloud, which is organized in a source-before-target order: (src_1, ..., src_B, tgt_1, ..., tgt_B).
            The name of fused feature tensor is "feats".
        2. This collation function build a graph pyramid with grid subsampling and radius neighbor searching. The keys
            "points", "lengths", "neighbors", "subsampling" and "upsampling" are five `list` holding the information of
            the graph pyramid.
        3. The correspondence indices are within each point cloud without accumulation over the batch.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        collated_dict = registration_collate_fn_pack_mode(data_dicts, unpack_single=self.unpack_single)

        # build graph pyramid
        points = np.concatenate([collated_dict.pop("src_points"), collated_dict.pop("tgt_points")], axis=0)
        lengths = np.concatenate([collated_dict.pop("src_lengths"), collated_dict.pop("tgt_lengths")], axis=0)
        graph_pyramid_dict = build_grid_pyramid_with_radius_graph_pack_mode(
            points, lengths, self.num_stages, self.voxel_size, self.search_radius, self.neighbor_limits
        )
        collated_dict.update(graph_pyramid_dict)

        # pack feats
        if "src_feats" in collated_dict and "tgt_feats" in collated_dict:
            feats = np.concatenate([collated_dict.pop("src_feats"), collated_dict.pop("tgt_feats")], axis=0)
            collated_dict["feats"] = feats

        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class RegistrationCollateFnPackModeWithUniformizedGraphPyramid(CollateFnPackModeWithUniformizedGraphPyramid):
    """Collate function for registration in pack mode with uniformized graph pyramid.

    WARNING: Under development.

    Note:
        1. The data of keys "src_points", "tgt_points", "src_feats" and "tgt_feats" are packed into large tensors by
            stacking along axis 0. The source point clouds and the target point clouds are then merged into a fused
            point cloud, which is organized in a source-before-target order: (src_1, ..., src_B, tgt_1, ..., tgt_B).
            The name of fused feature tensor is "feats".
        2. This collation function build a graph pyramid with grid subsampling and radius neighbor searching. The keys
            "points", "lengths", "neighbors", "subsampling" and "upsampling" are five `list` holding the information of
            the graph pyramid.
        3. The correspondence indices are within each point cloud without accumulation over the batch.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        collated_dict = registration_collate_fn_pack_mode(data_dicts, unpack_single=self.unpack_single)

        # build graph pyramid
        points = np.concatenate([collated_dict.pop("src_points"), collated_dict.pop("tgt_points")], axis=0)
        lengths = np.concatenate([collated_dict.pop("src_lengths"), collated_dict.pop("tgt_lengths")], axis=0)
        graph_pyramid_dict = build_uniformized_grid_pyramid_with_radius_graph_pack_mode(
            points,
            lengths,
            self.num_stages,
            self.voxel_size,
            self.search_radius,
            self.neighbor_limits,
            self.num_neighbors,
            self.distance_limit,
        )
        collated_dict.update(graph_pyramid_dict)

        scale = collated_dict["scale"]
        transform = collated_dict.pop("transform")
        transform = scale_transform(transform, scale)
        collated_dict["transform"] = transform.astype(np.float32)

        # pack feats
        if "src_feats" in collated_dict and "tgt_feats" in collated_dict:
            feats = np.concatenate([collated_dict.pop("src_feats"), collated_dict.pop("tgt_feats")], axis=0)
            collated_dict["feats"] = feats

        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class RegistrationCollateFnPackModeWithKnnGraphPyramid(CollateFnPackModeWithKnnGraphPyramid):
    """Collate function for registration in pack mode with graph pyramid.

    Note:
        1. The data of keys "src_points", "tgt_points", "src_feats" and "tgt_feats" are packed into large tensors by
            stacking along axis 0. The source point clouds and the target point clouds are then merged into a fused
            point cloud, which is organized in a source-before-target order: (src_1, ..., src_B, tgt_1, ..., tgt_B).
            The name of fused feature tensor is "feats".
        2. This collation function build a graph pyramid with grid subsampling and radius neighbor searching. The keys
            "points", "lengths", "neighbors", "subsampling" and "upsampling" are five `list` holding the information of
            the graph pyramid.
        3. The correspondence indices are within each point cloud without accumulation over the batch.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        collated_dict = registration_collate_fn_pack_mode(data_dicts, unpack_single=self.unpack_single)

        # build graph pyramid
        points = np.concatenate([collated_dict.pop("src_points"), collated_dict.pop("tgt_points")], axis=0)
        lengths = np.concatenate([collated_dict.pop("src_lengths"), collated_dict.pop("tgt_lengths")], axis=0)
        graph_pyramid_dict = build_grid_pyramid_with_knn_graph_pack_mode(
            points, lengths, self.num_stages, self.voxel_size, self.num_neighbors
        )
        collated_dict.update(graph_pyramid_dict)

        # pack feats
        if "src_feats" in collated_dict and "tgt_feats" in collated_dict:
            feats = np.concatenate([collated_dict.pop("src_feats"), collated_dict.pop("tgt_feats")], axis=0)
            collated_dict["feats"] = feats

        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class RgbdRegistrationCollateFnPackMode(CollateFnBase):
    """Simple RGB-D registration collate function in pack mode.

    For color and depth images, collate them as a batch.
    For back-projected point clouds, collate them as a pack.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        # 1. collate dict
        collated_dict = registration_collate_fn_pack_mode(data_dicts, unpack_single=False)

        # 2. stack RGB-D images
        src_color_img = np.stack(collated_dict.pop("src_color_img"), axis=0)
        src_depth_img = np.stack(collated_dict.pop("src_depth_img"), axis=0)
        tgt_color_img = np.stack(collated_dict.pop("tgt_color_img"), axis=0)
        tgt_depth_img = np.stack(collated_dict.pop("tgt_depth_img"), axis=0)

        collated_dict["src_color_img"] = src_color_img
        collated_dict["src_depth_img"] = src_depth_img
        collated_dict["tgt_color_img"] = tgt_color_img
        collated_dict["tgt_depth_img"] = tgt_depth_img

        # 3. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class RgbdRegistrationCollateFnPackModeWithGraphPyramid(CollateFnPackModeWithGraphPyramid):
    def __init__(
        self, num_stages: int, voxel_size: float, search_radius: float, neighbor_limits: list[int] | int | None = None
    ):
        super().__init__(num_stages, voxel_size, search_radius, neighbor_limits=neighbor_limits, unpack_single=False)

    def __call__(self, data_dicts: list[dict]) -> dict:
        # 1. collate dict
        collated_dict = registration_collate_fn_pack_mode(data_dicts, unpack_single=False)

        # 2. stack RGB-D images
        src_color_img = np.stack(collated_dict.pop("src_color_img"), axis=0)
        src_depth_img = np.stack(collated_dict.pop("src_depth_img"), axis=0)
        tgt_color_img = np.stack(collated_dict.pop("tgt_color_img"), axis=0)
        tgt_depth_img = np.stack(collated_dict.pop("tgt_depth_img"), axis=0)

        collated_dict["src_color_img"] = src_color_img
        collated_dict["src_depth_img"] = src_depth_img
        collated_dict["tgt_color_img"] = tgt_color_img
        collated_dict["tgt_depth_img"] = tgt_depth_img

        # 3. build graph pyramid
        points = np.concatenate([collated_dict.pop("src_points"), collated_dict.pop("tgt_points")], axis=0)
        lengths = np.concatenate([collated_dict.pop("src_lengths"), collated_dict.pop("tgt_lengths")], axis=0)
        graph_pyramid_dict = build_grid_pyramid_with_radius_graph_pack_mode(
            points, lengths, self.num_stages, self.voxel_size, self.search_radius, self.neighbor_limits
        )
        collated_dict.update(graph_pyramid_dict)

        # 3. pack feats
        if "src_feats" in collated_dict and "tgt_feats" in collated_dict:
            feats = np.concatenate([collated_dict.pop("src_feats"), collated_dict.pop("tgt_feats")], axis=0)
            collated_dict["feats"] = feats

        # 4. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class MultiModalRegistrationCollateFnPackModeWithGraphPyramid(CollateFnPackModeWithGraphPyramid):
    def __call__(self, data_dicts: list[dict]) -> dict:
        batch_size = len(data_dicts)

        # 1. collate dict
        collated_dict = collate_dict(data_dicts)

        # 2. handle batch size
        image = np.stack(collated_dict.pop("image"), axis=0)  # (B, *, H, W)
        depth = np.stack(collated_dict.pop("depth"), axis=0)  # (B, H, W)

        if batch_size == 1 and self.unpack_single:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["lengths"] = np.asarray([collated_dict["points"].shape[0]])
        else:
            points_list = collated_dict.pop("points")
            collated_dict["points"] = np.concatenate(points_list, axis=0)
            collated_dict["lengths"] = np.asarray([points.shape[0] for points in points_list])
            collated_dict["feats"] = np.concatenate(collated_dict.pop("feats"), axis=0)
            collated_dict["intrinsics"] = np.stack(collated_dict.pop("intrinsics"), axis=0)  # (B, 3, 3)

        collated_dict["image"] = image
        collated_dict["depth"] = depth

        collated_dict["batch_size"] = batch_size

        # 3. build graph pyramid
        points = collated_dict.pop("points")
        lengths = collated_dict.pop("lengths")
        graph_pyramid_dict = build_grid_pyramid_with_radius_graph_pack_mode(
            points, lengths, self.num_stages, self.voxel_size, self.search_radius, self.neighbor_limits
        )
        collated_dict.update(graph_pyramid_dict)

        # 4. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict


class NeRFCollateFn(CollateFnBase):
    """NeRF collate function.

    Simply concatenate everything along the first dimension and convert array to tensor.
    """

    def __call__(self, data_dicts: list[dict]) -> dict:
        if len(data_dicts) == 1:
            data_dict = array_to_tensor(data_dicts[0])
            return data_dict

        collated_dict = collate_dict(data_dicts)
        new_data_dict = {}
        for key, value in collated_dict.items():
            if isinstance(value[0], ndarray):
                new_data_dict[key] = np.concatenate(value, axis=0)
            elif isinstance(value[0], Tensor):
                new_data_dict[key] = torch.cat(value, dim=0)
            else:
                new_data_dict[key] = value
        new_data_dict = array_to_tensor(new_data_dict)
        return new_data_dict
