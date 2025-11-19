from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    use_conf: bool
    alpha: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        compare_target: bool = True,
        weight: str = "",
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"] if compare_target else batch["context"]["image"]
        delta = prediction.color - image
        dist = delta ** 2

        if self.cfg.use_conf:
            conf = prediction.conf.clamp(min=1.)
            dist = conf * dist - self.cfg.alpha * conf.log()

        return self.cfg.weight * dist.mean()
