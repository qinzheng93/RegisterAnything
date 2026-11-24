import torch
from torch import Tensor

from .index_select import index_select
from .knn import knn
from .pairwise_distance import pairwise_distance


@torch.no_grad()
def point_to_node_partition(
    points: Tensor,
    nodes: Tensor,
    point_limit: int | None = None,
    return_counts: bool = False,
    gather_points: bool = True,
    inf: float = 1e12,
) -> dict:
    """Point-to-Node partition to the point cloud.

    Assign each point to its nearest node. This is the single point cloud version. This function
    returns a dict containing the results. The key of the node indices each point assigned to is
    "point_to_node".

    If the "return_count" flag is True, this function will return the number of points assigned to
    each node. The key of this result is "node_sizes".

    If the "gather_points" flag is True, the points assigned to each node are collected. If there are
    too many points assigned to the same nodes, only the nearest "point_limit" points are reserved.
    There are three additional keys if "gather_points" is True:
    1. "node_masks": whether there are points assigned to a node. If True, there are points assigned
       to the node.
    2. "node_knn_indices": the indices of the points assigned to each node. The points are sorted from
       near to far. If certain entry is invalid, e.g., there are not enough points assigned to the node
       or the point is not assigned to the node, the entry is filled with `points.shape[0]`.
    3. "node_knn_masks": whether each entry in "node_knn_indices" is valid. If True, the entry is valid.

    Args:
        points (Tensor): the points to be partitioned with shape (N, 3).
        nodes (Tensor): the nodes with shape (M, 3).
        point_limit (int, optional): The max number of points to each node.
        return_count (bool): If True, return the sizes of the nodes. Default: False.
        gather_points (bool): If True, gather points for each node. Default: True.
        inf (float): A safe number for infinity. Default: 1e12.

    Returns:
        A dict with five entries at most:
        1. "point_to_node": the partition results.
        2. "node_sizes": the sizes of the nodes.
        3. "node_masks": whether there are points assigned to a node.
        4. "node_knn_indices": the indices of the points assigned to each node.
        5. "node_knn_masks": whether each entry in "node_knn_indices" is valid.
    """
    result_dict = {}

    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    result_dict["point_to_node"] = point_to_node

    if return_counts:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        result_dict["node_sizes"] = node_sizes

    if gather_points:
        node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
        node_masks.index_fill_(0, point_to_node, True)
        result_dict["node_masks"] = node_masks

        matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
        point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
        matching_masks[point_to_node, point_indices] = True  # (M, N)
        sq_dist_mat.masked_fill_(~matching_masks, inf)  # (M, N)

        max_points_per_node = int(matching_masks.sum(dim=1).max().item())
        if point_limit is not None:
            max_points_per_node = min(max_points_per_node, point_limit)
        assert max_points_per_node > 0, "All nodes are empty."

        node_knn_indices = sq_dist_mat.topk(k=max_points_per_node, dim=1, largest=False)[1]  # (M, K)
        node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
        node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, max_points_per_node)  # (M, K)
        node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
        node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])
        result_dict["node_knn_indices"] = node_knn_indices
        result_dict["node_knn_masks"] = node_knn_masks

    return result_dict


@torch.no_grad()
def ball_query_partition(
    points: Tensor,
    nodes: Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
) -> tuple[Tensor, ...]:
    node_knn_distances, node_knn_indices = knn(nodes, points, point_limit, return_distance=True)
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    node_knn_indices[~node_knn_masks] = points.shape[0]
    node_sizes = node_knn_masks.sum(1)  # (N,)
    node_masks = torch.gt(node_sizes, 0)  # (N,)

    if return_count:
        return node_knn_indices, node_knn_masks, node_masks, node_sizes

    return node_knn_indices, node_knn_masks, node_masks
