import warnings
from typing import Literal, overload

from numpy import ndarray
from scipy.spatial import KDTree

from vision3d.utils.misc import load_ext

ext_module = load_ext("vision3d.ext_cpu", ["knn_neighbors"])


@overload
def knn(q_points: ndarray, s_points: ndarray, k: int, return_distance: Literal[True]) -> tuple[ndarray, ndarray]: ...


@overload
def knn(q_points: ndarray, s_points: ndarray, k: int, return_distance: Literal[False] = False) -> ndarray: ...


def knn(
    q_points: ndarray, s_points: ndarray, k: int, return_distance: bool = False
) -> ndarray | tuple[ndarray, ndarray]:
    """Compute the nearest neighbor for the query points in support points.

    Note:
        If k=1, the return arrays are squeezed.
    """
    s_tree = KDTree(s_points)
    knn_distances, knn_indices = s_tree.query(q_points, k=k)

    if return_distance:
        return knn_distances, knn_indices  # type: ignore

    return knn_indices  # type: ignore


def knn_pack_mode(q_points: ndarray, s_points: ndarray, q_lengths: ndarray, s_lengths: ndarray, k: int) -> ndarray:
    """Computes neighbors for a batch of q_points and s_points, apply kNN search (in pack mode).

    This function is implemented on CPU.

    Args:
        q_points (ndarray): the query points (N, 3)
        s_points (ndarray): the support points (M, 3)
        q_lengths (ndarray): the list of lengths of batch elements in q_points
        s_lengths (ndarray): the list of lengths of batch elements in s_points
        k (int): the number of neighbors

    Returns:
        neighbors (ndarray): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    min_s_points = s_lengths.min()
    if k > min_s_points:
        warnings.warn(f"k={k} is larger than the number of support points {min_s_points}.")
    neighbor_indices: ndarray = ext_module.knn_neighbors(q_points, s_points, q_lengths, s_lengths, k)
    return neighbor_indices
