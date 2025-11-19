from dataclasses import dataclass

from jaxtyping import Float, Integer
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

from pytorch3d.loss import chamfer_distance


@dataclass
class LossChamferCfg:
    weight: float
    single_directional: bool


@dataclass
class LossChamferCfgWrapper:
    chamfer: LossChamferCfg


class LossChamfer(Loss[LossChamferCfg, LossChamferCfgWrapper]):
    def pad(
        self,
        gaussians: Gaussians,
    ) -> tuple[
        Float[Tensor, "batch n 3"],
        Integer[Tensor, "batch"]
    ]:
        B = gaussians.opacities.shape[0]
        max_len = 0
        for b in range(B):
            max_len = torch.sum(gaussians.opacities[b] > 1e-5)

        means = gaussians.means.new_zeros(B, max_len, 3)
        NP = torch.zeros([B], device=means.device, dtype=torch.long)
        for b in range(B):
            valid_means = gaussians.means[b][gaussians.opacities[b] > 1e-5]
            means[b, :valid_means.shape[0]] = valid_means
            NP[b] = valid_means.shape[0]
        return means, NP

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        b = batch["context"]["image"].shape[0]
        means = gaussians.means
        indices = gaussians.idx

        loss = 0.
        for i in range(b):
            means_single = means[i]
            indices_single = indices[i, ..., 0]
            loss += chamfer_distance(
                means_single[indices_single > 0].unsqueeze(0),
                means_single[indices_single == 0].unsqueeze(0).detach(),
                single_directional=self.cfg.single_directional,
            )[0]
        loss /= b
        return self.cfg.weight * loss
