from dataclasses import dataclass

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossPts3DCfg:
    weight: float


@dataclass
class LossPts3DCfgWrapper:
    pts3d: LossPts3DCfg


class LossPts3D(Loss[LossPts3DCfg, LossPts3DCfgWrapper]):
    def __init__(self, cfg: LossPts3DCfgWrapper):
        super().__init__(cfg)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt_lbs_weights = rearrange(batch["context"]["context_lbs_weights"], "b v h w c -> b (v h w) c")
        valid_mask = torch.sum(gt_lbs_weights, dim=-1) > 0

        gt_pts3d = rearrange(batch["context"]["context_pts3d"], "b v h w c -> b (v h w) c")
        pred_pts3d = gaussians.means

        loss = ((gt_pts3d - pred_pts3d) ** 2).sum(-1)
        loss = torch.sum(loss * valid_mask, dim=-1) / torch.sum(valid_mask, dim=-1)

        return self.cfg.weight * loss.mean()
