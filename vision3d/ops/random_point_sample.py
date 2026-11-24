import torch
from torch import Tensor

from .gather import gather


def random_point_sample(points: Tensor, num_samples: int, gather_points: bool = True) -> Tensor:
    batch_size, _, num_points = points.shape
    weights = torch.ones(size=(batch_size, num_points)).cuda()
    indices = torch.multinomial(weights, num_samples)

    if not gather_points:
        return indices

    samples = gather(points, indices)
    return samples
