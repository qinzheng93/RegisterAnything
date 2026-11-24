from typing import Literal, overload

import torch
from torch import Tensor


def _get_indices_from_lengths(lengths: Tensor, num_items: int) -> Tensor:
    """Compute the indices in flattened batch tensor from the lengths in pack mode."""
    length_list = lengths.detach().cpu().numpy().tolist()
    chunks = [(i * num_items, i * num_items + length) for i, length in enumerate(length_list)]
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    return indices


@overload
def batch_to_pack(
    batch_tensor: Tensor, *, masks: Tensor | None = None, lengths: Tensor | None = None, return_lengths: Literal[True]
) -> tuple[Tensor, Tensor]: ...


@overload
def batch_to_pack(
    batch_tensor: Tensor,
    *,
    masks: Tensor | None = None,
    lengths: Tensor | None = None,
    return_lengths: Literal[False] = False,
) -> Tensor: ...


def batch_to_pack(
    batch_tensor: Tensor, *, masks: Tensor | None = None, lengths: Tensor | None = None, return_lengths: bool = True
) -> Tensor | tuple[Tensor, Tensor]:
    """Convert Tensor from batch mode to stack mode with masks.

    Args:
        batch_tensor (Tensor): the input tensor in batch mode (B, N, *).
        masks (Tensor[bool]): the masks of items of each sample in the batch (B, N).
        return_lengths (bool): whether to return the lengths of each sample in the batch. Default: True.

    Returns:
        A Tensor in pack mode with shape (M, *).
        A Tensor[long] of the length of each sample in the batch with shape (B).
    """
    if masks is not None and lengths is not None:
        raise ValueError("masks and lengths cannot be specified at the same time.")

    if masks is not None:
        pack_tensor = batch_tensor[masks]
    else:
        batch_size = batch_tensor.shape[0]
        num_items = batch_tensor.shape[1]
        new_shape = (batch_size * num_items,) + batch_tensor.shape[2:]
        batch_tensor = batch_tensor.view(*new_shape)
        if lengths is not None:
            src_indices = _get_indices_from_lengths(lengths, num_items)
            pack_tensor = batch_tensor[src_indices]
        else:
            pack_tensor = batch_tensor.view(*new_shape)

    if not return_lengths:
        return pack_tensor

    if lengths is not None:
        return pack_tensor, lengths

    if masks is not None:
        lengths = masks.sum(dim=1)
    else:
        lengths = torch.full(size=(batch_tensor.shape[0],), fill_value=batch_tensor.shape[1], dtype=torch.long).cuda()

    return pack_tensor, lengths


@overload
def pack_to_batch(
    pack_tensor: Tensor,
    lengths: Tensor,
    *,
    max_length: int | None = None,
    fill_value: float = 0.0,
    return_masks: Literal[True],
) -> tuple[Tensor, Tensor]: ...


@overload
def pack_to_batch(
    pack_tensor: Tensor,
    lengths: Tensor,
    *,
    max_length: int | None = None,
    fill_value: float = 0.0,
    return_masks: Literal[False] = False,
) -> Tensor: ...


def pack_to_batch(
    pack_tensor: Tensor,
    lengths: Tensor,
    *,
    max_length: int | None = None,
    fill_value: float = 0.0,
    return_masks: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    """Convert Tensor from pack mode to batch mode.

    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (Tensor[long]): The number of items of each sample in the batch (B)
        max_length (int, optional): The maximal length of each sample in the batch.
        fill_value (float|int|bool): The default value in the empty regions. Default: 0.
        return_masks (bool): Whether to return the masks of each sample in the batch. Default: True.

    Returns:
        A Tensor in stack mode with shape (B, N, C), where N is max(lengths).
        A Tensor[bool] of the masks of each sample in the batch with shape (B, N).
    """
    batch_size = lengths.shape[0]
    if max_length is None:
        max_length = int(lengths.max().item())
    tgt_indices = _get_indices_from_lengths(lengths, max_length)

    num_channels = pack_tensor.shape[1]
    batch_tensor = pack_tensor.new_full(size=(batch_size * max_length, num_channels), fill_value=fill_value)
    batch_tensor[tgt_indices] = pack_tensor
    batch_tensor = batch_tensor.view(batch_size, max_length, num_channels)

    if not return_masks:
        return batch_tensor

    masks = torch.zeros(size=(batch_size * max_length,), dtype=torch.bool).cuda()
    masks[tgt_indices] = True
    masks = masks.view(batch_size, max_length)

    return batch_tensor, masks
