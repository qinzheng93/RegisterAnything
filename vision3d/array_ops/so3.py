from typing import overload

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation


def get_rotation_along_axis(scale: float, axis: str = "z") -> ndarray:
    assert axis in ["x", "y", "z"]
    theta = 2.0 * np.pi * scale
    if axis == "x":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]]
        )
    elif axis == "y":
        rotation = np.asarray(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 1.0], [-np.sin(theta), 0.0, np.cos(theta)]]
        )
    else:
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
    return rotation


@overload
def apply_rotation(points: ndarray, rotation: ndarray, normals: None = None) -> ndarray: ...


@overload
def apply_rotation(points: ndarray, rotation: ndarray, normals: ndarray) -> tuple[ndarray, ndarray]: ...


def apply_rotation(
    points: ndarray, rotation: ndarray, normals: ndarray | None = None
) -> ndarray | tuple[ndarray, ndarray]:
    new_points = np.matmul(points, rotation.T)
    if normals is None:
        return new_points
    new_normals = np.matmul(normals, rotation.T)
    return new_points, new_normals


def skew_symmetric_matrix(vector: ndarray) -> ndarray:
    """Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Args:
        vector (ndarray): input vectors (3)

    Returns:
        skews (ndarray): output skew-symmetric matrix (3, 3)
    """
    skews = np.zeros(shape=(3, 3))
    skews[0, 1] = -vector[2]
    skews[0, 2] = vector[1]
    skews[1, 0] = vector[2]
    skews[1, 2] = -vector[0]
    skews[2, 0] = -vector[1]
    skews[2, 1] = vector[0]
    return skews


def rodrigues_rotation_formula(axis: ndarray, angle: float) -> ndarray:
    """Compute Rodrigues Rotation Matrix.

    R = I + sin{theta} K + (1 - cos{theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Args:
        axis (array<float>): normalized axis vectors (3)
        angle (float): rotation angles in right-hand direction in rad.

    Returns:
        rotation (array<float>): Rodrigues rotation matrix (3, 3)
    """
    skews = skew_symmetric_matrix(axis)  # (3, 3)
    rotation = np.eye(3) + np.sin(angle) * skews + (1.0 - np.cos(angle)) * np.matmul(skews, skews)
    return rotation


def axis_angle_to_rotation_matrix(phi: ndarray) -> ndarray:
    rotation = Rotation.from_rotvec(phi).as_matrix()
    return rotation


def axis_angle_to_quaternion(phi: ndarray) -> ndarray:
    q = Rotation.from_rotvec(phi).as_quat()
    q = q[..., [3, 0, 1, 2]]  # (wxyz) -> (xyzw)
    return q


def quaternion_to_axis_angle(q: ndarray) -> ndarray:
    q = q[..., [1, 2, 3, 0]]  # (wxyz) -> (xyzw)
    phi = Rotation.from_quat(q).as_rotvec()
    return phi


def quaternion_to_rotation_matrix(q: ndarray) -> ndarray:
    q = q[..., [1, 2, 3, 0]]  # (wxyz) -> (xyzw)
    rotation = Rotation.from_quat(q).as_matrix()
    return rotation


def rotation_matrix_to_axis_angle(rotation: ndarray) -> ndarray:
    phi = Rotation.from_matrix(rotation).as_rotvec()
    return phi


def rotation_matrix_to_quaternion(rotation: ndarray) -> ndarray:
    q = Rotation.from_matrix(rotation).as_quat()
    q = q[..., [3, 0, 1, 2]]  # (xyzw) -> (wxyz)
    return q


def rotation_matrix_to_euler(rotation: ndarray, order: str, use_degree: bool = False) -> ndarray:
    euler = Rotation.from_matrix(rotation).as_euler(order, degrees=use_degree)
    return euler


def euler_to_rotation_matrix(euler: ndarray, order: str) -> ndarray:
    rotation = Rotation.from_euler(order, euler).as_matrix()
    return rotation


def random_sample_rotation(rotation_factor: float = 1.0) -> ndarray:
    """Random sample a rotation matrix for strong/standard data augmentation.

    Steps:
        1. random sample three euler angles from [0, 2pi / factor].
        2. compute the rotation matrix from the euler angles.
    """
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = euler_to_rotation_matrix(euler, "zyx")
    return rotation


def random_sample_rotation_norm() -> ndarray:
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis) + 1e-8
    theta = np.pi * np.random.rand()
    euler = axis * theta
    rotation = euler_to_rotation_matrix(euler, "zyx")
    return rotation


def get_alignment_rotation_matrix(src_vector: ndarray, tgt_vector: ndarray, eps: float = 1e-5) -> ndarray:
    """Compute the rotation matrix aligning the source vector to the target vector using Rodrigues' Rotation Formula.

    Args:
        src_vector (ndarray): The source vectors (3)
        tgt_vector (ndarray): The target vectors (3)
        eps (float=1e-5): A safe number.

    Returns:
        rotation (ndarray): rotation matrix (3, 3)
    """
    # check norm
    src_normal = np.linalg.norm(src_vector)  # (3)
    tgt_normal = np.linalg.norm(tgt_vector)  # (3)
    assert src_normal > eps, "Zero src_vector."
    assert tgt_normal > eps, "Zero tgt_vector."

    # normalize
    src_vector = src_vector / src_normal  # (3)
    tgt_vector = tgt_vector / tgt_normal  # (3)

    # compute axis
    src_skew = skew_symmetric_matrix(src_vector)  # (3, 3)
    phi = np.matmul(src_skew, tgt_vector)  # (3)

    # compute angle
    cos_value = (src_vector * tgt_vector).sum().clip(min=-1.0, max=1.0)

    if cos_value > 1.0 - eps:
        # parallel case: no rotation needed.
        return np.eye(3)

    if cos_value < -1.0 + eps:
        # opposite case: 180 degrees rotation around arbitrary axis.
        aux_vector = np.zeros(3)  # (3)
        if np.abs(src_vector[0]) < 0.9:
            aux_vector[0] = 1.0  # (1, 0, 0)
        else:
            aux_vector[1] = 1.0  # (0, 1, 0)
        phi = np.matmul(src_skew, aux_vector)  # (3)
        theta = np.pi  # set norm to pi
    else:
        theta = np.arccos(cos_value)  # (3)

    phi = phi / np.linalg.norm(phi) * theta  # (3)

    # compute rodrigues formula
    rotation = axis_angle_to_rotation_matrix(phi)

    return rotation
