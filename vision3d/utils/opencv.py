import cv2
import numpy as np
from numpy import ndarray

from vision3d.array_ops.se3 import get_transform_from_rotation_translation
from vision3d.array_ops.so3 import axis_angle_to_rotation_matrix


def registration_with_pnp_ransac(
    corr_points: ndarray,
    corr_pixels: ndarray,
    intrinsics: ndarray,
    distortion: ndarray | None = None,
    num_iterations: int = 5000,
    distance_tolerance: float = 8.0,
    transposed: bool = True,
) -> ndarray | None:
    """PnP-RANSAC registration with OpenCV.

    Note:
        1. cv2.solvePnPRansac() requires the pixels are in the order of (w, h).

    Args:
        corr_points (array): a float array of the 3D correspondence points with shape (N, 3).
        corr_pixels (array): an int array of the 2D correspondence pixels with shape (N, 2).
        intrinsics (array): a float array of the camera intrinsics with shape (3, 3).
        distortion (array, optional): a float array of the distortion parameter with shape (4, 1) or (12, 1).
        num_iterations (int): the number of ransac iterations.
        distance_tolerance (float): the distance tolerance for ransac.
        transposed (bool): if True, the pixel coordinates are in the order of (h, w) or (w, h) otherwise.

    Returns:
        A float array of the estimated transformation from 3D to 2D.
    """
    if corr_points.shape[0] < 4:
        # too few correspondences, return None
        return None

    if distortion is None:
        distortion = np.zeros(shape=(4, 1))
    if transposed:
        corr_pixels = np.stack([corr_pixels[..., 1], corr_pixels[..., 0]], axis=-1)  # (h, w) -> (w, h)

    corr_points = corr_points.astype(np.float32)
    corr_pixels = corr_pixels.astype(np.float32)
    intrinsics = intrinsics.astype(np.float32)

    _, axis_angle, translation, _ = cv2.solvePnPRansac(
        corr_points,
        corr_pixels,
        intrinsics,
        distortion,
        iterationsCount=num_iterations,
        reprojectionError=distance_tolerance,
        flags=cv2.SOLVEPNP_P3P,
    )
    axis_angle = axis_angle[:, 0]
    translation = translation[:, 0]
    rotation = axis_angle_to_rotation_matrix(axis_angle)
    estimated_transform = get_transform_from_rotation_translation(rotation, translation)

    return estimated_transform
