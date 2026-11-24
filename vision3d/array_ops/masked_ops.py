import numpy as np
from numpy import ndarray


def masked_sum(inputs: ndarray, masks: ndarray, transposed: bool = False, keepdims: bool = False) -> ndarray:
    """Masked sum.

    Args:
        inputs (ndarray): The input tensor with shape (B, N, C).
        masks (ndarray): The input masks with shape (B, N).
        transposed (bool): If True, the input shape if (B, C, N). Default: False.
        keepdims (bool): If True, keep the reduced dimension. Default: False.

    Returns:
        A ndarray of the masked sum with shape (B, C) if keepdims is False, else (B, 1, C).
    """
    # convert (B, C, N) to (B, N, C) if transposed
    if transposed:
        inputs = inputs.transpose(0, 2, 1)  # (B, N, C)

    # compute masked sum
    masks = masks[..., np.newaxis]  # (B, N, 1)
    sum_values = (inputs * masks).sum(axis=1, keepdims=keepdims)  # (B, C) or (B, 1, C)

    # convert (B, 1, C) to (B, C, 1) if transposed and keepdims
    if keepdims and transposed:
        sum_values = sum_values.transpose(0, 2, 1)  # (B, C, 1)

    return sum_values


def masked_mean(
    inputs: ndarray, masks: ndarray, transposed: bool = False, keepdims: bool = False, eps: float = 1e-6
) -> ndarray:
    """Masked mean.

    Args:
        inputs (ndarray): The input tensor with shape (B, N, C).
        masks (ndarray): The input masks with shape (B, N).
        transposed (bool): If True, the input shape if (B, C, N). Default: False.
        keepdims (bool): If True, keep the reduced dimension. Default: False.
        eps (float): A safe number for division. Default: 1e-6.

    Returns:
        A ndarray of the masked mean with shape (B, C) or (B, 1, C) if keepdims is True.
    """
    # convert (B, C, N) to (B, N, C) if transposed
    if transposed:
        inputs = inputs.transpose(0, 2, 1)  # (B, N, C)

    # compute masked mean
    masks = masks[..., np.newaxis]  # (B, N, 1)
    sum_values = (inputs * masks).sum(axis=1, keepdims=keepdims)  # (B, C) or (B, 1, C)
    counts = masks.sum(axis=1, keepdims=keepdims)  # (B, 1) or (B, 1, 1)
    mean_values = sum_values / (counts + eps)  # (B, C) or (B, 1, C)

    # convert (B, 1, C) to (B, C, 1) if transposed and keepdims
    if keepdims and transposed:
        mean_values = mean_values.transpose(0, 2, 1)  # (B, C, 1)

    return mean_values


def masked_normalize(inputs: ndarray, masks: ndarray, transposed: bool = False) -> ndarray:
    """Masked normalize.

    Keep the input values for the masked out entries.

    Args:
        inputs (ndarray): The input tensor with shape (B, N, C).
        masks (ndarray): The input masks with shape (B, N).
        transposed (bool): If True, the input shape if (B, C, N). Default: False.

    Returns:
        A ndarray of the masked normalized outputs with shape (B, N, C).
    """
    # convert (B, C, N) to (B, N, C) if transposed
    if transposed:
        inputs = inputs.transpose(0, 2, 1)  # (B, N, C)

    # compute masked normalize
    mean_values = masked_mean(inputs, masks, keepdims=True)  # (B, 1, C)
    normalized_values = inputs - mean_values  # (B, N, C)
    normalized_values = np.where(masks[..., np.newaxis], normalized_values, inputs)  # (B, N, C)

    # convert (B, 1, C) to (B, C, 1) if transposed and keepdims
    if transposed:
        normalized_values = normalized_values.transpose(0, 2, 1)

    return normalized_values
