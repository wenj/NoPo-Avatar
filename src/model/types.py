from dataclasses import dataclass, fields

from typing import Optional
from jaxtyping import Float
import torch
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    lbs_weights: Optional[Float[Tensor, "batch gaussian dim"]] = None
    lbs_weights_bones: Optional[Float[Tensor, "batch gaussian dim2"]] = None
    idx: Optional[Float[Tensor, "batch gaussian 3"]] = None
    nums: Optional[list] = None
    uvs: Optional[Float[Tensor, "batch gaussian dim"]] = None
    conf: Float[Tensor, "batch gaussian"] | None = None
    opacities_hard: Optional[Float[Tensor, "batch gaussian"]] = None
    noise: Optional[Float[Tensor, "batch gaussian_t dim"]] = None

    def clone(self):
        return Gaussians(**{
            f.name: getattr(self, f.name).clone() if isinstance(getattr(self, f.name), torch.Tensor)
            else getattr(self, f.name)
            for f in fields(self)
        })
