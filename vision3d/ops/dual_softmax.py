import torch
from torch import Tensor


def dual_softmax(score_mat: Tensor, row_masks: Tensor | None = None, col_masks: Tensor | None = None) -> Tensor:
    """Dual softmax.

    output = softmax(input, dim=-1) * softmax(input, dim=-2)

    Args:
        score_mat (Tensor): the score matrix with shape (*, N, M).
        row_masks (Tensor[bool], optional): the masks for the rows with shape (*, N).
        col_masks (Tensor[bool], optional): the masks for the columns with shape (*, M).

    Returns:
        A Tensor of the dual softmax score matrix with shape (*, N, M).
    """
    output_mat = torch.softmax(score_mat, dim=-1) * torch.softmax(score_mat, dim=-2)
    if row_masks is not None:
        output_mat = torch.where(row_masks.unsqueeze(-1), output_mat, torch.zeros_like(output_mat))
    if col_masks is not None:
        output_mat = torch.where(col_masks.unsqueeze(-2), output_mat, torch.zeros_like(output_mat))
    return output_mat
