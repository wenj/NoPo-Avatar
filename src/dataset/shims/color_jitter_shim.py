import torch
from jaxtyping import Float
from torch import Tensor
import torchvision.transforms.functional as TF

from ..types import AnyExample, AnyViews


def jitter_view(
    views: AnyViews,
    brightness_factor: Float,
    contrast_factor: Float,
    saturation_factor: Float,
    hue_factor: Float
) -> AnyViews:
    img_jittered = TF.adjust_brightness(views["image"], brightness_factor)
    img_jittered = TF.adjust_contrast(img_jittered, contrast_factor)
    img_jittered = TF.adjust_saturation(img_jittered, saturation_factor)
    img_jittered = TF.adjust_hue(img_jittered, hue_factor)
    img_jittered = img_jittered * views["mask"].unsqueeze(1) + views["image"] * (1 - views["mask"].unsqueeze(1))
    return {
        **views,
        "image": img_jittered,
    }


def apply_color_jitter_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""

    brightness_factor = 1 + (torch.rand(tuple(), generator=generator) * 0.8 - 0.4)
    contrast_factor = 1 + (torch.rand(tuple(), generator=generator) * 0.8 - 0.4)
    saturation_factor = 1 + (torch.rand(tuple(), generator=generator) * 0.8 - 0.4)
    hue_factor = torch.rand(tuple(), generator=generator) * 0.8 - 0.4


    return {
        **example,
        "context": jitter_view(example["context"], brightness_factor, contrast_factor, saturation_factor, hue_factor),
        "target": jitter_view(example["target"], brightness_factor, contrast_factor, saturation_factor, hue_factor),
    }