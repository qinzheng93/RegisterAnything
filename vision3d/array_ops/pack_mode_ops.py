import numpy as np
from numpy import ndarray


def mean_pack_mode(packed_array: ndarray, lengths: ndarray) -> ndarray:
    """Compute the mean value of the packed array along the given axis.

    Args:
        packed_array (ndarray): The data array in pack mode, with shape (total_N, *).
        lengths (ndarray): The length of each sample in the batch, with shape (B,).

    Returns:
        ndarray: The mean value of the packed array, with shape (B, *).
    """
    batch_size = lengths.shape[0]
    mean_values_list = []
    s_index = 0
    for batch_index in range(batch_size):
        e_index = s_index + lengths[batch_index]
        cur_array = packed_array[s_index:e_index]
        cur_mean_value = cur_array.mean(axis=0)
        mean_values_list.append(cur_mean_value)
        s_index = e_index
    mean_values = np.stack(mean_values_list, axis=0)
    return mean_values


def multiply_pack_mode(packed_array: ndarray, lengths: ndarray, scales: ndarray) -> ndarray:
    """Multiply the packed array with the scale factor.

    Args:
        packed_array (ndarray): The data array in pack mode, with shape (total_N, *).
        lengths (ndarray): The length of each sample in the batch, with shape (B,).
        scales (ndarray): The scale factor of each sample, with shape (B, *).

    Returns:
        ndarray: The scaled array, with shape (total_N, *).
    """
    batch_size = lengths.shape[0]
    scaled_array_list = []
    s_index = 0
    for batch_index in range(batch_size):
        e_index = s_index + lengths[batch_index]
        cur_array = packed_array[s_index:e_index]
        cur_scaled_array = cur_array * scales[batch_index]
        scaled_array_list.append(cur_scaled_array)
        s_index = e_index
    scaled_array = np.concatenate(scaled_array_list, axis=0)
    return scaled_array
