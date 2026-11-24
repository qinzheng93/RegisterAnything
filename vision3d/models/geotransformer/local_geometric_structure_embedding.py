import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers.embedding import SinusoidalEmbedding
from vision3d.ops.index_select import index_select
from vision3d.ops.pairwise_distance import pairwise_distance


@torch.no_grad()
def get_local_pairwise_distance_inputs(
    points: Tensor,
    neighbor_indices: Tensor,
    sigma_d: float,
    dist_mat: Tensor | None = None,
) -> Tensor:
    """Compute the inputs of pair-wise distance embedding.

    Args:
        points: a Tensor of the input point cloud with shape (B, N, 3).
        neighbor_indices: a Tensor of the neighbor indices with shape (B, N, k).
        sigma_d: a float of the distance embedding sigma.
        dist_mat (optional): a Tensor of the precomputed distance matrix with shape (B, N, N).

    Returns:
        A Tensor of the distance embedding inputs with shape (B, N, k).
    """
    _, num_points, _ = points.shape
    assert num_points > 1, "Too few superpoints."

    # distance map
    if dist_mat is None:
        dist_mat = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

    neighbor_dist_mat = torch.gather(dist_mat, 2, neighbor_indices)  # (B, N, k)

    # distance inputs
    d_inputs = neighbor_dist_mat / sigma_d  # (B, N, k)

    return d_inputs


class LocalPairwiseDistanceEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, sigma: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.sigma = sigma

        self.embedding = SinusoidalEmbedding(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points: Tensor, neighbor_indices: Tensor, dist_mat: Tensor | None = None) -> Tensor:
        """Local Pairwise Distance Embedding forward.

        Args:
            points: a Tensor of the input point cloud with shape (B, N, 3).
            neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).
            dist_mat (optional): a Tensor of the precomputed distance matrix with shape (B, N, N).

        Returns:
            A Tensor of the distance embedding with shape (B, N, K, C).
        """
        inputs = get_local_pairwise_distance_inputs(
            points, neighbor_indices, self.sigma, dist_mat=dist_mat
        )  # (B, N, k)

        embeddings = self.embedding(inputs)  # (B, N, K, C)
        embeddings = self.proj(embeddings)  # (B, N, K, C)

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
            f"sigma={self.sigma:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


@torch.no_grad()
def get_local_tripletwise_angular_inputs(
    points: Tensor,
    neighbor_indices: Tensor,
    sigma: float,
    k: int,
    dist_mat: Tensor | None = None,
) -> Tensor:
    """Compute the inputs of triplet-wise angular embedding.

    Args:
        points: a Tensor of the input point cloud with shape (B, N, 3).
        neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).
        sigma: a float of the angular embedding sigma.
        k: an int of the number of nearest neighbors.
        dist_mat (optional): a Tensor of the precomputed distance matrix with shape (B, N, N).

    Returns:
        A Tensor of the angular embedding inputs with shape (B, N, K, A).
    """
    factor = 180.0 / (sigma * np.pi)
    batch_size, num_points, _ = points.shape
    num_neighbors = neighbor_indices.shape[2]
    assert num_points > 1, "Too few superpoints."

    # distance map
    if dist_mat is None:
        dist_mat = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

    # get neighbors
    exp_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)  # (B, N, N, 3)
    neighbor_indices = neighbor_indices.unsqueeze(3).expand(batch_size, num_points, num_neighbors, 3)  # (B, N, K, 3)
    neighbor_points = torch.gather(exp_points, dim=2, index=neighbor_indices)  # (B, N, K, 3)

    # angular inputs
    k = min(k, num_points - 1)
    knn_indices = dist_mat.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, A)
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_points, k, 3)  # (B, N, A, 3)
    knn_points = torch.gather(exp_points, dim=2, index=knn_indices)  # (B, N, A, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, A, 3)
    anc_vectors = neighbor_points - points.unsqueeze(2)  # (B, N, K, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_points, num_neighbors, k, 3)  # (B, N, K, A, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_points, num_neighbors, k, 3)  # (B, N, K, A, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, K, A)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, K, A)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, K, A)
    a_inputs = angles * factor  # (B, N, K, A)

    return a_inputs


class LocalTripletwiseAngularEmbedding(nn.Module):
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

    def forward(self, points: Tensor, neighbor_indices: Tensor, dist_mat: Tensor | None = None) -> Tensor:
        """Local Tripletwise Angular Embedding forward.

        Args:
            points: a Tensor of the input point cloud with shape (B, N, 3).
            neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).
            dist_mat (optional): a Tensor of the precomputed distance matrix with shape (B, N, N).

        Returns:
            A Tensor of the distance embedding with shape (B, N, K, C).
        """
        inputs = get_local_tripletwise_angular_inputs(points, neighbor_indices, self.sigma, self.k, dist_mat=dist_mat)

        embeddings = self.embedding(inputs)  # (B, N, K, A, C)
        embeddings = self.proj(embeddings)  # (B, N, K, A, C)
        if self.reduction == "max":
            if self.training:
                # use torch.max to improve training speed
                embeddings = embeddings.max(dim=3)[0]  # (B, N, K, C)
            else:
                # use torch.amax to reduce memory footprint during testing
                embeddings = embeddings.amax(dim=3)  # (B, N, K, C)
        else:
            embeddings = embeddings.mean(dim=3)  # (B, N, K, C)

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


class LocalGeometricStructureEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sigma_d: float,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction_a: str = "max",
    ):
        super().__init__()

        self.d_embedding = LocalPairwiseDistanceEmbedding(hidden_dim, sigma_d)
        self.a_embedding = LocalTripletwiseAngularEmbedding(hidden_dim, sigma_a, angle_k, reduction=reduction_a)

    def forward(self, points: Tensor, neighbor_indices: Tensor) -> Tensor:
        """Local Geometric Structure Embedding forward.

        Args:
            points: a Tensor of the input point cloud with shape (B, N, 3).
            neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).

        Returns:
            A Tensor of the distance embedding with shape (B, N, K, C).
        """
        dist_mat = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_embeddings = self.d_embedding(points, neighbor_indices, dist_mat=dist_mat)  # (B, N, K, C)
        a_embeddings = self.a_embedding(points, neighbor_indices, dist_mat=dist_mat)  # (B, N, K, C)
        embeddings = d_embeddings + a_embeddings  # (B, N, K, C)
        return embeddings


@torch.no_grad()
def get_local_tripletwise_angular_inputs_v2(
    points: Tensor, neighbor_indices: Tensor, k: int, dist_mat: Tensor | None = None
) -> Tensor:
    """Compute the inputs of triplet-wise angular embedding.

    Compared to V1, this version discretizes the angles into degree grids (totally 181 grids) with rounding function.
    To this end,

    Args:
        points: a Tensor of the input point cloud with shape (B, N, 3).
        neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).
        k: an int of the number of nearest neighbors.
        dist_mat (optional): a Tensor of the precomputed distance matrix with shape (B, N, N).

    Returns:
        A Tensor of the angular embedding inputs with shape (B, N, K, A).
    """
    batch_size, num_points, _ = points.shape
    num_neighbors = neighbor_indices.shape[2]
    assert num_points > 1, "Too few superpoints."

    # distance map
    if dist_mat is None:
        dist_mat = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

    # get neighbors
    exp_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)  # (B, N, N, 3)
    neighbor_indices = neighbor_indices.unsqueeze(3).expand(batch_size, num_points, num_neighbors, 3)  # (B, N, K, 3)
    neighbor_points = torch.gather(exp_points, dim=2, index=neighbor_indices)  # (B, N, K, 3)

    # angular indices
    k = min(k, num_points - 1)
    knn_indices = dist_mat.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, A)
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_points, k, 3)  # (B, N, A, 3)
    knn_points = torch.gather(exp_points, dim=2, index=knn_indices)  # (B, N, A, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, A, 3)
    anc_vectors = neighbor_points - points.unsqueeze(2)  # (B, N, K, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_points, num_neighbors, k, 3)  # (B, N, K, A, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_points, num_neighbors, k, 3)  # (B, N, K, A, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, K, A)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, K, A)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, K, A)
    a_inputs = torch.round(angles * (180.0 / np.pi)).long()  # (B, N, K, A), in degrees

    return a_inputs


class LocalTripletwiseAngularEmbeddingV2(nn.Module):
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

    def forward(self, points: Tensor, neighbor_indices: Tensor, dist_mat: Tensor | None = None) -> Tensor:
        """Local Triplet-wise Angular Embedding V2 forward.

        Args:
            points: a Tensor of the input point cloud with shape (B, N, 3).
            neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).
            dist_mat (optional): a Tensor of the precomputed distance matrix with shape (B, N, N).

        Returns:
            A Tensor of the angular embedding with shape (B, N, K, C).
        """
        indices = get_local_tripletwise_angular_inputs_v2(
            points, neighbor_indices, self.k, dist_mat=dist_mat
        )  # (B, N, K, A)

        # angular embeddings
        all_inputs = torch.arange(181).cuda().float() / self.sigma  # (181)
        all_embeddings = self.embedding(all_inputs)  # (181, C)
        all_embeddings = self.proj(all_embeddings)  # (181, C)
        embeddings = index_select(all_embeddings, indices, dim=0)  # (B, N, K, A, C)
        if self.reduction == "max":
            if self.training:
                embeddings = embeddings.max(dim=3)[0]  # (B, N, K, C)
            else:
                embeddings = embeddings.amax(dim=3)  # (B, N, K, C)
        else:
            embeddings = embeddings.mean(dim=3)  # (B, N, K, C)

        return embeddings  # (B, N, K, C)

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
            f"sigma={self.sigma:g}",
            f"k={self.k}",
            f"reduction={self.reduction}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


class LocalGeometricStructureEmbeddingV2(nn.Module):
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

        self.d_embedding = LocalPairwiseDistanceEmbedding(hidden_dim, sigma_d)
        self.a_embedding = LocalTripletwiseAngularEmbeddingV2(hidden_dim, sigma_a, angle_k, reduction=reduction_a)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.proj_e = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, points: Tensor, neighbor_indices: Tensor) -> Tensor:
        """Geometric Structure Embedding V2 forward.

        Args:
            points: a Tensor of the input point cloud with shape (B, N, 3).
            neighbor_indices: a Tensor of the neighbor indices with shape (B, N, K).

        Returns:
            A Tensor of the geometric structure embedding with shape (B, N, K, C).
        """
        # distance matrix
        dist_mat = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)

        # PDE & TAE
        d_embeddings = self.d_embedding(points, neighbor_indices, dist_mat=dist_mat)  # (B, N, K, C)
        a_embeddings = self.a_embedding(points, neighbor_indices, dist_mat=dist_mat)  # (B, N, K, C)

        # geometric structure embeddings
        embeddings = d_embeddings + a_embeddings  # (B, N, K, C)
        embeddings = self.act(embeddings)  # (B, N, K, C)
        embeddings = self.proj_e(embeddings)  # (B, N, K, C)
        embeddings = self.norm(embeddings)  # (B, N, K, C)

        return embeddings  # (B, N, K, C)

    def extra_repr(self) -> str:
        param_strings = [
            f"hidden_dim={self.hidden_dim}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
