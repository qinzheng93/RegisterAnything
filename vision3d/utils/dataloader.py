import random
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from .collate import CollateFnPackModeWithGraphPyramid
from .distributed import is_distributed
from .misc import deprecated
from .types import CollateFn

# Pack mode utilities


@deprecated("calibrate_neighbors_pack_mode_v2")
def calibrate_neighbors_pack_mode_deprecated(
    dataset: Dataset,
    collate_fn_class: Any,  # class
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    keep_ratio: float = 0.8,
    sample_threshold: int = 2000,
) -> list[int]:
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    collate_fn = collate_fn_class(num_stages, voxel_size, search_radius, max_neighbor_limits)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):  # type: ignore
        data_dict = collate_fn([dataset[i]])

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict["neighbors"]]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0).tolist()

    return neighbor_limits


def calibrate_neighbors_pack_mode(
    dataset: Dataset,
    collate_fn: CollateFnPackModeWithGraphPyramid,
    keep_ratio: float = 0.8,
    sample_threshold: int | None = 2000,
) -> list[int]:
    # initialize histogram of neighborhood sizes
    num_stages = collate_fn.num_stages
    max_neighbors = np.max(collate_fn.neighbor_limits).item()
    neighbor_hists = np.zeros((num_stages, max_neighbors), dtype=np.int64)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):  # type: ignore
        data_dict = collate_fn([dataset[i]])

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict["neighbors"]]
        hists = [np.bincount(c, minlength=max_neighbors)[:max_neighbors] for c in counts]
        neighbor_hists += np.vstack(hists)

        if sample_threshold is not None and np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[max_neighbors - 1, :]), axis=0).tolist()

    return neighbor_limits


# Common dataloader


def reset_seed_worker_init_fn(worker_id: int) -> None:
    """Reset NumPy and Python seed for data loader worker."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 1,
    shuffle: bool | None = None,
    collate_fn: CollateFn | None = None,
    sampler: Sampler | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    if is_distributed():
        if sampler is None:
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            warnings.warn("Custom sampler is used in DDP mode. Make sure your sampler correctly handles DDP sampling.")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader
