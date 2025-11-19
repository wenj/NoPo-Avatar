from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerUniformCfg:
    name: Literal["uniform"]
    num_context_views: int
    num_target_views: int


class ViewSamplerUniform(ViewSampler[ViewSamplerUniformCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape

        interval = num_views // self.cfg.num_context_views

        index_context = [torch.randint(
            num_views,
            size=tuple(),
            device=device,
        ).item()]
        for _ in range(self.cfg.num_context_views - 1):
            next_interval = torch.randint(interval, interval + 2, size=tuple(), device=device).item()
            index_context.append((index_context[-1] + next_interval) % num_views)

        index_target = torch.randint(
            num_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        return (
            torch.tensor(index_context),
            index_target,
            overlap
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views