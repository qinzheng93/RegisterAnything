import torch
from torch import Tensor

from .conversion import batch_to_pack, pack_to_batch


def voxelize_pack_mode(
    points: Tensor, lengths: Tensor, voxel_size: float, feats: Tensor | None = None
) -> tuple[Tensor, ...]:
    """Voxelize point cloud and reduce features (optional) in pack mode.

    Args:
        points (Tensor): stack of points. (N, 3)
        lengths (LongTensor): length of each point cloud. (B,)
        voxel_size (float): voxel size
        feats (Tensor=None): stack of features. (N, C)

    Returns:
        nodes (Tensor): stack of nodes. (M, 3)
        vlengths (LongTensor): length of each node cloud. (B,)
        vfeats (Tensor): stack of node features. (M, 3)
    """
    batch_size = lengths.shape[0]

    voxels = torch.floor(points / voxel_size).long()  # (N, 3)
    min_voxel = voxels.min(dim=0)[0]  # (3,)
    max_voxel = voxels.max(dim=0)[0]  # (3,)
    voxel_dim = max_voxel - min_voxel + 1  # (3,)
    voxels = voxels - min_voxel.unsqueeze(0)  # (N, 3)

    base0 = voxel_dim[0] * voxel_dim[1] * voxel_dim[2]
    base1 = voxel_dim[1] * voxel_dim[2]
    base2 = voxel_dim[2]

    length_list: list[int] = lengths.detach().cpu().numpy().tolist()
    b_indices = torch.cat(
        [torch.full(size=(length_list[i],), fill_value=i, dtype=torch.long) for i in range(batch_size)], dim=0
    ).cuda()  # (N,)
    b_voxels = torch.cat([b_indices.unsqueeze(1), voxels], dim=1)  # (N, 4)
    b_values = b_voxels[:, 0] * base0 + b_voxels[:, 1] * base1 + b_voxels[:, 2] * base2 + b_voxels[:, 3]  # (N,)
    u_values, inv_indices, u_counts = torch.unique(b_values, return_inverse=True, return_counts=True)  # (M,)

    inv_indices = inv_indices.unsqueeze(1)  # (M, 1)
    u_counts = u_counts.unsqueeze(1).float()  # (M, 1)

    voxels = torch.stack([u_values % base0 // base1, u_values % base1 // base2, u_values % base2], dim=1)
    voxels = voxels + min_voxel.unsqueeze(0)

    nodes = torch.zeros(size=(u_values.shape[0], 3)).cuda()  # (M, 3)
    n_inv_indices = inv_indices.expand(-1, 3)  # (M, 3)
    nodes.scatter_add_(dim=0, index=n_inv_indices, src=points)  # (M, 3)
    nodes /= u_counts  # (M, 3)

    b_indices = u_values // base0  # (M,)
    v_lengths = torch.zeros(size=(batch_size,), dtype=torch.long).cuda()  # (B)
    ones = torch.ones_like(b_indices, dtype=torch.long).cuda()  # (M)
    v_lengths.scatter_add_(dim=0, index=b_indices, src=ones)

    if feats is None:
        return voxels, nodes, v_lengths

    v_feats = torch.zeros(size=(voxels.shape[0], feats.shape[1])).cuda()  # (M, C)
    f_inv_indices = inv_indices.expand(-1, feats.shape[1])  # (N, C)
    v_feats.scatter_add_(dim=0, index=f_inv_indices, src=feats)  # (M, C)
    v_feats /= u_counts  # (M, C)
    return voxels, nodes, v_feats, v_lengths


def voxelize(
    points: Tensor, voxel_size: float, masks: Tensor | None = None, feats: Tensor | None = None
) -> tuple[Tensor, ...]:
    """Voxelize point cloud and reduce features (optional) in batch mode.

    Args:
        points (Tensor): the points in batch mode (B, N, 3).
        voxel_size (float): voxel size
        feats (Tensor=None): the features in batch mode (B, N, C).
        masks (Tensor[bool]): the masks of points in the batch (B,).

    Returns:
        voxels (LongTensor): batch of voxels. (B, M, 3)
        nodes (Tensor): batch of nodes. (B, M, 3)
        feats (Tensor): batch of node features. (B, M, C)
        masks (Tensor[bool]): If False, the voxel does not exist in that sample. (B, M)
    """
    points, lengths = batch_to_pack(points, masks=masks)

    if feats is not None:
        feats = batch_to_pack(feats, masks=masks, return_lengths=False)
        voxels, nodes, v_feats, v_lengths = voxelize_pack_mode(points, lengths, voxel_size, feats=feats)
        voxels, v_masks = pack_to_batch(voxels, v_lengths)
        nodes = pack_to_batch(nodes, v_lengths, return_masks=False)
        v_feats = pack_to_batch(v_feats, v_lengths, return_masks=False)
        return voxels, nodes, v_feats, v_masks

    voxels, nodes, v_lengths = voxelize_pack_mode(points, lengths, voxel_size)
    voxels, v_masks = pack_to_batch(voxels, v_lengths)
    nodes = pack_to_batch(nodes, v_lengths, return_masks=False)
    return voxels, nodes, v_masks
