from numpy import ndarray

from vision3d.utils.misc import load_ext

ext_module = load_ext("vision3d.ext_cpu", ["radius_neighbors"])


def radius_search_pack_mode(
    q_points: ndarray, s_points: ndarray, q_lengths: ndarray, s_lengths: ndarray, radius: float, neighbor_limit: int
) -> ndarray:
    """Computes neighbors for a batch of q_points and s_points, apply radius search (in pack mode).

    This function is implemented on CPU.

    Args:
        q_points (ndarray): the query points (N, 3)
        s_points (ndarray): the support points (M, 3)
        q_lengths (ndarray): the list of lengths of batch elements in q_points
        s_lengths (ndarray): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (ndarray): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    neighbor_indices: ndarray = ext_module.radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius)
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices
