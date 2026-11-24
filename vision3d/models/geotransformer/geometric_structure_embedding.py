import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers.embedding import SinusoidalEmbedding
from vision3d.ops.index_select import index_select
from vision3d.ops.pairwise_distance import pairwise_distance


@torch.no_grad()
def get_pairwise_distance_inputs(points: Tensor, sigma_d: float, dist_map: Tensor | None = None) -> Tensor:
    """Compute the inputs of pair-wise distance embedding.

    Args:
        points: a Tensor of the input point cloud with shape (B, N, 3).
        sigma_d: a float of the distance embedding sigma.
        dist_map (optional): a Tensor of the precomputed distance map with shape (B, N, N).

    Returns:
        A Tensor of the distance embedding inputs with shape (B, N, N).
    """
    _, num_points, _ = points.shape
    assert num_points > 1, "Too few superpoints."

    # distance map
    if dist_map is None:
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

    # distance inputs
    d_inputs = dist_map / sigma_d

    return d_inputs


class PairwiseDistanceEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, sigma: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.sigma = sigma

        self.embedding = SinusoidalEmbedding(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points: Tensor, dist_map: Tensor | None = None) -> Tensor:
        inputs = get_pairwise_distance_inputs(points, self.sigma, dist_map=dist_map)

        embeddings = self.embedding(inputs)
        embeddings = self.proj(embeddings)

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
            f"sigma={self.sigma:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


@torch.no_grad()
def get_tripletwise_angular_inputs(points: Tensor, sigma: float, k: int, dist_map: Tensor | None = None) -> Tensor:
    """Compute the inputs of triplet-wise angular embedding.

    Args:
        points: a Tensor of the input point cloud with shape (B, N, 3).
        sigma: a float of the angular embedding sigma.
        k: an int of the number of nearest neighbors.
        dist_map (optional): a Tensor of the precomputed distance map with shape (B, N, N).

    Returns:
        A Tensor of the angular embedding inputs with shape (B, N, N, k).
    """
    factor = 180.0 / (sigma * np.pi)
    batch_size, num_points, _ = points.shape
    assert num_points > 1, "Too few superpoints."

    # distance map
    if dist_map is None:
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

    # angular inputs
    k = min(k, num_points - 1)
    knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_points, k, 3)  # (B, N, k, 3)
    exp_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)  # (B, N, N, 3)
    knn_points = torch.gather(exp_points, dim=2, index=knn_indices)  # (B, N, k, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
    anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_points, num_points, k, 3)  # (B, N, N, k, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_points, num_points, k, 3)  # (B, N, N, k, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
    a_inputs = angles * factor

    return a_inputs


class TripletwiseAngularEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sigma: float = 15.0,
        k: int = 3,
        reduction: str = "max",
    ):
        super().__init__()

        assert reduction in ["max", "mean"], f"Unsupported reduction mode: {reduction}."

        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.k = k
        self.reduction = reduction

        self.embedding = SinusoidalEmbedding(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points: Tensor, dist_map: Tensor | None = None) -> Tensor:
        inputs = get_tripletwise_angular_inputs(points, self.sigma, self.k, dist_map=dist_map)

        embeddings = self.embedding(inputs)
        embeddings = self.proj(embeddings)
        if self.reduction == "max":
            if self.training:
                # use torch.max to improve training speed
                embeddings = embeddings.max(dim=3)[0]
            else:
                # use torch.amax to reduce memory footprint during testing
                embeddings = embeddings.amax(dim=3)
        else:
            embeddings = embeddings.mean(dim=3)

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
            f"sigma={self.sigma:g}",
            f"k={self.k}",
            f"reduction={self.reduction}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


class GeometricStructureEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sigma_d: float,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction_a: str = "max",
    ):
        super().__init__()

        self.d_embedding = PairwiseDistanceEmbedding(hidden_dim, sigma_d)
        self.a_embedding = TripletwiseAngularEmbedding(hidden_dim, sigma_a, angle_k, reduction=reduction_a)

    def forward(self, points: Tensor) -> Tensor:
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_embeddings = self.d_embedding(points, dist_map=dist_map)
        a_embeddings = self.a_embedding(points, dist_map=dist_map)
        embeddings = d_embeddings + a_embeddings
        return embeddings


@torch.no_grad()
def get_tripletwise_angular_inputs_v2(points: Tensor, k: int, dist_map: Tensor | None = None) -> Tensor:
    """Compute the inputs of triplet-wise angular embedding.

    Compared to V1, this version discretizes the angles into degree bins (totally 181 bins) with rounding function.
    To this end, it reduces the computation of triplet-wise angular embedding into a projection on 181 bins and then
    a indexing into (B, N, N, C).

    Args:
        points: a Tensor of the input point cloud with shape (B, N, 3).
        k: an int of the number of nearest neighbors.
        dist_map (optional): a Tensor of the precomputed distance map with shape (B, N, N).

    Returns:
        A Tensor of the angular embedding inputs with shape (B, N, N, k).
    """
    batch_size, num_points, _ = points.shape
    assert num_points > 1, "Too few superpoints."

    # distance map
    if dist_map is None:
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

    # angular indices
    k = min(k, num_points - 1)
    knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_points, k, 3)  # (B, N, k, 3)
    exp_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)  # (B, N, N, 3)
    knn_points = torch.gather(exp_points, dim=2, index=knn_indices)  # (B, N, k, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
    anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_points, num_points, k, 3)  # (B, N, N, k, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_points, num_points, k, 3)  # (B, N, N, k, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
    a_inputs = torch.round(angles * (180.0 / np.pi)).long()  # in degrees

    return a_inputs


class TripletwiseAngularEmbeddingV2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sigma: float = 15.0,
        k: int = 3,
        reduction: str = "max",
    ):
        super().__init__()

        assert reduction in ["max", "mean"], f"Unsupported reduction mode: {reduction}."

        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.k = k
        self.reduction = reduction

        self.embedding = SinusoidalEmbedding(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points: Tensor, dist_map: Tensor | None = None) -> Tensor:
        indices = get_tripletwise_angular_inputs_v2(points, self.k, dist_map=dist_map)

        # angular embeddings
        all_inputs = torch.arange(181).cuda().float() / self.sigma  # (181)
        all_embeddings = self.embedding(all_inputs)  # (181, C)
        all_embeddings = self.proj(all_embeddings)  # (181, C)
        embeddings = index_select(all_embeddings, indices, dim=0)  # (B, N, N, k, C)
        if self.reduction == "max":
            if self.training:
                embeddings = embeddings.max(dim=3)[0]
            else:
                embeddings = embeddings.amax(dim=3)
        else:
            embeddings = embeddings.mean(dim=3)

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
            f"sigma={self.sigma:g}",
            f"k={self.k}",
            f"reduction={self.reduction}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


class GeometricStructureEmbeddingV2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sigma_d: float,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction_a: str = "max",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.d_embedding = PairwiseDistanceEmbedding(hidden_dim, sigma_d)
        self.a_embedding = TripletwiseAngularEmbeddingV2(hidden_dim, sigma_a, angle_k, reduction=reduction_a)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.proj_e = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, points: Tensor) -> Tensor:
        # distance map
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

        # PDE & TAE
        d_embeddings = self.d_embedding(points, dist_map=dist_map)  # (B, N, N, C)
        a_embeddings = self.a_embedding(points, dist_map=dist_map)  # (B, N, N, C)

        # geometric structure embeddings
        embeddings = d_embeddings + a_embeddings
        embeddings = self.act(embeddings)
        embeddings = self.proj_e(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
