import numpy as np
from numpy import ndarray

from .grid_subsample import grid_subsample_pack_mode
from .knn import knn_pack_mode
from .radius_search import radius_search_pack_mode


def uniformize_point_cloud_pack_mode(
    points: ndarray, lengths: ndarray, num_neighbors: int, distance_limit: float
) -> float:
    """Uniformize the density of the point cloud in pack mode.

    Strategy:
        Scale the point cloud so that the mean distance of the `k` nearest neighbor over all points equals to `radius`.

    Note:
        This function currently scale *ALL* points in the batch with the same scaling factor, so this function only support
        **the batch size of 1**. The reason is that this is more convenient for the processing of registration data, where
        the source and the target point clouds should be scaled with the same scaling factor. In this case, should we perform
        the uniformization in the collate_fn or the transform_fn?

    Args:
        points (ndarray): The point cloud in pack mode.
        lengths (ndarray): The lengths of the point cloud in pack mode.
        num_neighbors (int): k nearest neighbors.
        distance_limit (float): The target distance of the mean knn distance.

    Returns:
        A float of the scale factor.
    """
    neighbor_indices = knn_pack_mode(points, points, lengths, lengths, num_neighbors)  # (totN, k)
    neighbor_points = points[neighbor_indices]  # (totN, k, 3)
    neighbor_distances = np.linalg.norm(neighbor_points - points[:, np.newaxis, :], axis=2)  # (totN, k)
    mean_neighbor_distances = neighbor_distances.mean(axis=1)  # (totN,)
    # mean_distance = mean_neighbor_distances.mean().item()
    # scale = distance_limit / mean_distance  # (B,)
    median_distance = np.median(mean_neighbor_distances).item()
    scale = distance_limit / median_distance  # (B,)
    return scale


def build_grid_pyramid_pack_mode(
    points: ndarray,
    lengths: ndarray,
    num_stages: int,
    voxel_size: float,
) -> dict:
    """Build point pyramid with grid subsampling in pack mode."""
    assert num_stages is not None
    assert voxel_size is not None

    points_list = []
    lengths_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample_pack_mode(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    return {"points": points_list, "lengths": lengths_list}


def build_uniformized_grid_pyramid_pack_mode(
    points: ndarray,
    lengths: ndarray,
    num_stages: int,
    voxel_size: float,
    num_neighbors: int,
    distance_limit: float,
) -> dict:
    """Build uniformized point pyramid with grid subsampling in pack mode."""
    scale = uniformize_point_cloud_pack_mode(points, lengths, num_neighbors, distance_limit)
    points = points * scale
    point_dict = build_grid_pyramid_pack_mode(points, lengths, num_stages, voxel_size)
    point_dict["scale"] = scale
    return point_dict


def build_radius_graph_pack_mode(
    point_dict: dict,
    radius: float,
    neighbor_limits: list[int],
) -> dict:
    """Build graph pyramid with grid subsampling and radius searching in pack mode."""
    assert radius is not None
    assert neighbor_limits is not None

    points_list = point_dict["points"]
    lengths_list = point_dict["lengths"]
    num_stages = len(points_list)

    assert num_stages == len(neighbor_limits)

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        # compute neighbors
        neighbors = radius_search_pack_mode(
            cur_points, cur_points, cur_lengths, cur_lengths, radius, neighbor_limits[i]
        )
        neighbors_list.append(neighbors)

        if i == num_stages - 1:
            break

        sub_points = points_list[i + 1]
        sub_lengths = lengths_list[i + 1]

        # compute subsampling
        subsampling = radius_search_pack_mode(
            sub_points, cur_points, sub_lengths, cur_lengths, radius, neighbor_limits[i]
        )
        subsampling_list.append(subsampling)

        # compute upsampling
        upsampling = radius_search_pack_mode(
            cur_points, sub_points, cur_lengths, sub_lengths, radius * 2, neighbor_limits[i + 1]
        )
        upsampling_list.append(upsampling)

        radius *= 2

    return {"neighbors": neighbors_list, "subsampling": subsampling_list, "upsampling": upsampling_list}


def build_knn_graph_pack_mode(
    point_dict: dict,
    num_neighbors: int,
) -> dict:
    """Build graph pyramid with grid subsampling and radius searching in pack mode."""
    points_list = point_dict["points"]
    lengths_list = point_dict["lengths"]
    num_stages = len(points_list)

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # knn search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        # compute neighbors
        neighbors = knn_pack_mode(cur_points, cur_points, cur_lengths, cur_lengths, num_neighbors)
        neighbors_list.append(neighbors)

        if i == num_stages - 1:
            break

        sub_points = points_list[i + 1]
        sub_lengths = lengths_list[i + 1]

        # compute subsampling
        subsampling = knn_pack_mode(sub_points, cur_points, sub_lengths, cur_lengths, num_neighbors)
        subsampling_list.append(subsampling)

        # compute upsampling
        upsampling = knn_pack_mode(cur_points, sub_points, cur_lengths, sub_lengths, num_neighbors)
        upsampling_list.append(upsampling)

    return {"neighbors": neighbors_list, "subsampling": subsampling_list, "upsampling": upsampling_list}


def build_grid_pyramid_with_radius_graph_pack_mode(
    points: ndarray,
    lengths: ndarray,
    num_stages: int,
    voxel_size: float,
    radius: float,
    neighbor_limits: list[int],
) -> dict:
    """Build graph pyramid with grid subsampling and radius searching in pack mode."""
    point_dict = build_grid_pyramid_pack_mode(points, lengths, num_stages, voxel_size)
    graph_dict = build_radius_graph_pack_mode(point_dict, radius, neighbor_limits)
    data_dict = point_dict | graph_dict
    return data_dict


def build_uniformized_grid_pyramid_with_radius_graph_pack_mode(
    points: ndarray,
    lengths: ndarray,
    num_stages: int,
    voxel_size: float,
    radius: float,
    neighbor_limits: list[int],
    num_neighbors: int,
    distance_limit: float,
) -> dict:
    """Build uniformized graph pyramid with grid subsampling and radius searching in pack mode."""
    point_dict = build_uniformized_grid_pyramid_pack_mode(
        points, lengths, num_stages, voxel_size, num_neighbors, distance_limit
    )
    graph_dict = build_radius_graph_pack_mode(point_dict, radius, neighbor_limits)
    data_dict = point_dict | graph_dict
    return data_dict


def build_grid_pyramid_with_knn_graph_pack_mode(
    points: ndarray,
    lengths: ndarray,
    num_stages: int,
    voxel_size: float,
    num_neighbors: int,
) -> dict:
    """Build graph pyramid with grid subsampling and radius searching in pack mode."""
    point_dict = build_grid_pyramid_pack_mode(points, lengths, num_stages, voxel_size)
    graph_dict = build_knn_graph_pack_mode(point_dict, num_neighbors)
    data_dict = point_dict | graph_dict
    return data_dict
