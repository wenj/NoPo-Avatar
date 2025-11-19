from dataclasses import dataclass
import numpy as np
import cv2

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

from ..misc.body_utils import apply_lbs_to_means


@dataclass
class LossProjection2Cfg:
    weight: float


@dataclass
class LossProjection2CfgWrapper:
    projection2: LossProjection2Cfg


class LossProjection2(Loss[LossProjection2Cfg, LossProjection2CfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        return_loss_map: bool = False,
    ):
        b, v, _, h, w = batch["context"]["image"].shape
        n = gaussians.means.shape[1]
        lbs_weights = F.softmax(gaussians.lbs_weights, dim=-1)

        view_indices = (gaussians.idx[..., 0] - 1).clamp(min=0)
        indices = gaussians.idx[..., 1:]
        u = gaussians.idx[..., 2] / w
        v = gaussians.idx[..., 1] / h

        means = gaussians.means
        extrinsics = torch.linalg.inv(batch["context"]["extrinsics"]) # B x V x 4 x 4
        extrinsics = extrinsics.gather(dim=1, index=view_indices[..., None, None].repeat(1, 1, *extrinsics.shape[2:])) # B x N x 4 x 4
        Rs = batch["context"]["Rs"].gather(dim=1, index=view_indices[..., None, None, None].repeat(1, 1, *batch["context"]["Rs"].shape[2:])) # B x N x J x 3 x 3
        Ts = batch["context"]["Ts"].gather(dim=1, index=view_indices[..., None, None].repeat(1, 1, *batch["context"]["Ts"].shape[2:])) # B x N x J x 3
        means_posed = apply_lbs_to_means(means.reshape(b * n, 1, 3), Rs.reshape(b * n, -1, 3, 3), Ts.reshape(b * n, -1, 3), lbs_weights.reshape(b * n, 1, -1))
        means_posed = means_posed.reshape(b, n, 3)
        means_cam = (extrinsics[..., :3, :3] @ means_posed.unsqueeze(-1)).squeeze(-1) + extrinsics[..., :3, 3]

        intrinsics = batch["context"]["intrinsics"].gather(dim=1, index=view_indices[..., None, None].repeat(1, 1, *batch["context"]["intrinsics"].shape[2:])) # B x N x 3 x 3
        fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
        cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
        pix = torch.stack([(u - cx) / fx.clamp(min=1e-5), (v - cy) / fy.clamp(min=1e-5), fx.new_ones(b, n)], dim=-1)
        dist = torch.cross(means_cam, pix).norm(dim=-1) / pix.norm(dim=-1)

        loss = (torch.sum(dist * (gaussians.idx[..., 0] > 0), dim=1) / torch.sum(gaussians.idx[..., 0] > 0, dim=1)).mean()

        return self.cfg.weight * loss
