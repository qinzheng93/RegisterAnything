import torch.nn as nn
from torch import Tensor

from vision3d.ops.sc2_registration import sc2_registration
from vision3d.ops.spectral_registration import spectral_registration


class SpectralRegistration(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, src_corr_points: Tensor, tgt_corr_points: Tensor, corr_masks: Tensor | None = None) -> Tensor:
        transform = spectral_registration(src_corr_points, tgt_corr_points, self.sigma, masks=corr_masks)
        return transform


class RegistrationSC2(nn.Module):
    def __init__(self, sigma: float, k: int = 10, tight_metric: bool = False):
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.tight_metric = tight_metric

    def forward(self, src_corr_points: Tensor, tgt_corr_points: Tensor, corr_masks: Tensor | None = None) -> Tensor:
        transform = sc2_registration(
            src_corr_points, tgt_corr_points, self.sigma, masks=corr_masks, k=self.k, tight_metric=self.tight_metric
        )
        return transform
