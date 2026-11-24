from typing import overload

import numpy as np
from numpy import ndarray

from vision3d.array_ops.point_cloud_utils import random_sample_direction
from vision3d.array_ops.so3 import euler_to_rotation_matrix, rodrigues_rotation_formula


@overload
def apply_transform(points: ndarray, transform: ndarray, normals: None = None) -> ndarray: ...


@overload
def apply_transform(points: ndarray, transform: ndarray, normals: ndarray) -> tuple[ndarray, ndarray]: ...


def apply_transform(
    points: ndarray, transform: ndarray, normals: ndarray | None = None
) -> ndarray | tuple[ndarray, ndarray]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    new_points = np.matmul(points, rotation.T) + translation

    if normals is None:
        return new_points

    new_normals = np.matmul(normals, rotation.T)
    return new_points, new_normals


def compose_transforms(*transforms: ndarray) -> ndarray:
    """
    Compose transforms from the first one to the last one.
    T = T_{n_1} circ T_{n_2} circ ... circ T_1 circ T_0
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = np.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation: ndarray | None, translation: ndarray | None) -> ndarray:
    """Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: ndarray) -> tuple[ndarray, ndarray]:
    """Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def inverse_transform(transform: ndarray) -> ndarray:
    """Inverse rigid transform.

    Args:
        transform (array): (4, 4)

    Return:
        inv_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (3, 3), (3,)
    inv_rotation = rotation.T  # (3, 3)
    inv_translation = -np.matmul(inv_rotation, translation)  # (3,)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (4, 4)
    return inv_transform


def scale_transform(transform: ndarray, scale: float) -> ndarray:
    """Scale transform.

    Args:
        transform (array): (4, 4)
        scaling_factor (float): Scaling factor.

    Returns:
        scaled_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)
    translation = translation * scale
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform


def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = euler_to_rotation_matrix(euler, "zyx")
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform


def random_sample_small_transform(scale: float = 0.1) -> ndarray:
    """Random sample a small transform for weak data augmentation.

    Steps:
        1. random sample an axis.
        2. random sample a rotation angle: [0, sigma * pi / sqrt(3)]
        3. compute the rotation matrix with Rodrigues' equation.
        4. random sample a translation vector: [-sigma / sqrt(3), sigma / sqrt(3)].
        5. compose transformation matrix.
    """
    axis = random_sample_direction()
    theta = np.random.rand() * scale * np.pi / np.sqrt(3.0)
    rotation = rodrigues_rotation_formula(axis, theta)
    translation = np.random.randn(3) * scale / np.sqrt(3.0)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform
