import numpy as np
from numpy import ndarray


def build_local_reference_frame(points: ndarray, normals: ndarray) -> tuple[ndarray, ...]:
    """Build Local Reference Frame.

    Refer to ``The Perfect Match: 3D Point Cloud Matching with Smoothed Densities``.

    Args:
        points (ndarray): (N, 3)
        normals (ndarray): (3,)
        return_axes (bool=False): If True, return the lrf axes.

    Returns:
        x_axes (ndarray): (3,)
        y_axes (ndarray): (3,)
        z_axes (ndarray): (3,)
        lrf_points (ndarray): (N, 3)
    """
    z_axes = normals / np.linalg.norm(normals)  # (3,)
    offsets = points - points.mean(axis=0, keepdims=True)  # (N, 3)
    distances = np.linalg.norm(offsets, axis=1)  # (N,)
    z_coords = np.matmul(offsets, z_axes)  # (N,)
    plane_offsets = offsets - z_coords[:, np.newaxis] * z_axes[np.newaxis]  # (N, 3)
    weights = np.abs(z_coords) / (distances + 1e-10)  # (N,)
    weights = weights / weights.sum()  # (N,)
    x_axes = (plane_offsets * weights[:, np.newaxis]).sum(axis=0)  # (3,)
    x_axes = x_axes / np.linalg.norm(x_axes)  # (3,)
    y_axes = np.cross(x_axes, z_axes)  # (3,)

    x_coords = np.matmul(offsets, x_axes)  # (N,)
    y_coords = np.matmul(offsets, y_axes)  # (N,)

    lrf_points = np.stack([x_coords, y_coords, z_coords], axis=1)  # (N, 3)

    return x_axes, y_axes, z_axes, lrf_points
