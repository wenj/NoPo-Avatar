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
class LossLBSWeightsCfg:
    weight: float


@dataclass
class LossLBSWeightsCfgWrapper:
    lbs_weights: LossLBSWeightsCfg


class LossLBSWeights(Loss[LossLBSWeightsCfg, LossLBSWeightsCfgWrapper]):
    def __init__(self, cfg: LossLBSWeightsCfgWrapper):
        super().__init__(cfg)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if "lbs_weights" not in batch["context"]:
            return torch.tensor(0.0, device=batch["context"]["mask"].device)

        b, v, h, w = batch["context"]["mask"].shape
        gt_lbs_weights = rearrange(batch["context"]["lbs_weights"], "b v h w c -> b (v h w) c")
        valid_mask = (torch.sum(gt_lbs_weights, dim=-1) > 0) * batch["context"]["mask"].reshape(b, -1)
        gt_lbs_weights = F.normalize(gt_lbs_weights.clip(min=0), dim=-1)
        pred_lbs_weights = gaussians.lbs_weights
        # import pdb; pdb.set_trace()
        # print('loss', pred_lbs_weights.isnan().any(), pred_lbs_weights.isinf().any(), gt_lbs_weights.isnan().any())

        loss = self.loss(
            rearrange(pred_lbs_weights, "b n c -> b c n"),
            rearrange(gt_lbs_weights, "b n c -> b c n"),
        )
        # print('loss', loss.isnan().any(), pred_lbs_weights.isnan().any(), gt_lbs_weights.isnan().any(), valid_mask.isnan().any(), torch.sum(valid_mask, dim=-1))
        # print('loss', loss.isinf().any(), pred_lbs_weights.isinf().any(), gt_lbs_weights.isinf().any(),
        #       valid_mask.isinf().any(), torch.sum(valid_mask, dim=-1))
        loss = torch.sum(loss * valid_mask, dim=-1) / (torch.sum(valid_mask, dim=-1) + 1e-5)

        return self.cfg.weight * loss.mean()
