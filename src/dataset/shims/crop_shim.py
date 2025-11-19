import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
import cv2
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torchvision.transforms import InterpolationMode
import random

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    mask: Float[Tensor, "h_in w_in"],
    lbs_weight: Float[Tensor, "h_in w_in d"] | None,
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    # image_new = cv2.resize(image_new, (w, h), interpolation=cv2.INTER_LANCZOS4)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    image_new = rearrange(image_new, "h w c -> c h w")

    mask_new = (mask * 255).clip(min=0, max=255).type(torch.uint8)
    mask_new = mask_new.detach().cpu().numpy()
    mask_new = Image.fromarray(mask_new)
    mask_new = mask_new.resize((w, h), Image.NEAREST)
    # mask_new = cv2.resize(mask_new, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_new = np.array(mask_new) / 255
    mask_new = torch.tensor(mask_new, dtype=mask.dtype, device=mask.device)

    if lbs_weight is not None:
        if lbs_weight.shape[:2] == shape:
            lbs_weight_new = lbs_weight
        else:
            lbs_weight_new = F.interpolate(lbs_weight[None].permute(0, 3, 1, 2), (h, w), mode='bilinear', antialias=True)
            lbs_weight_new = lbs_weight_new.permute(0, 2, 3, 1)[0]
    else:
        lbs_weight_new = None
    return image_new, mask_new, lbs_weight_new


def center_pad(
    images: Float[Tensor, "*#batch c h w"],
    masks: Float[Tensor, "*#batch h w"],
    lbs_weights: Float[Tensor, "*#batch h w d"] | None,
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    bgcolor: Float[Tensor, "*#batch 3"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch h_out w_out"],  # updated masks
    Float[Tensor, "*#batch h_out w_out d"] | None,  # updated lbs_weights
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_out - h_in) // 2
    col = (w_out - w_in) // 2

    # Center-crop the image.
    if bgcolor is not None:
        images_pad = repeat(bgcolor, "... -> n ... h_out w_out", n=images.shape[0], h_out=h_out, w_out=w_out).clone()
        images_pad[..., row:row + h_in, col:col + w_in] = images
        images = images_pad
    else:
        images = F.pad(images, (col, w_out - w_in - col, row, h_out - h_in - row), "replicate")
    masks = F.pad(masks, (col, w_out - w_in - col, row, h_out - h_in - row))
    if lbs_weights is not None:
        if lbs_weights.shape[:2] != shape:
            lbs_weights = F.pad(lbs_weights, (0, 0, col, w_out - w_in - col, row, h_out - h_in - row))

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * w_in / w_out + col / w_out
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * h_in / h_out + row / h_out

    # return images, masks, lbs_weights, intrinsics
    return images, masks, lbs_weights, intrinsics


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    masks: Float[Tensor, "*#batch h w"],
    lbs_weights: Float[Tensor, "*#batch h w d"] | None,
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch h_out w_out"],  # updated masks
    Float[Tensor, "*#batch h_out w_out d"] | None,  # updated lbs_weights
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]
    masks = masks[..., :, row : row + h_out, col : col + w_out]
    if lbs_weights is not None:
        if lbs_weights.shape[:2] != shape:
            lbs_weights = lbs_weights[..., row : row + h_out, col : col + w_out, :]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * w_in / w_out - col / w_out
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * h_in / h_out - row / h_out

    # return images, masks, lbs_weights, intrinsics
    return images, masks, lbs_weights, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    masks: Float[Tensor, "*#batch h w"],
    lbs_weights: Float[Tensor, "*#batch h w d"] | None,
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    pad: bool = False,
    bgcolor: Float[Tensor, "*#batch 3"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch h_out w_out"],  # updated masks
    Float[Tensor, "*#batch h w d"] | None, # updated lbs_weights
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    if h_out <= h_in and w_out <= w_in:
        assert h_out <= h_in and w_out <= w_in

        if pad:
            scale_factor = min(h_out / h_in, w_out / w_in)
        else:
            scale_factor = max(h_out / h_in, w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        assert h_scaled == h_out or w_scaled == w_out

        # Reshape the images to the correct size. Assume we don't have to worry about
        # changing the intrinsics based on how the images are rounded.
        *batch, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        masks = masks.reshape(-1, h, w)
        if lbs_weights is not None:
            d = lbs_weights.shape[-1]
            lbs_weights = lbs_weights.reshape(-1, h, w, d)
        else:
            lbs_weights = [None] * images.shape[0]
        images_new, masks_new, lbs_weights_new = [], [], []
        for (image, mask, lbs_weight) in zip(images, masks, lbs_weights):
            image_new, mask_new, lbs_weight_new = rescale(image, mask, lbs_weight, (h_scaled, w_scaled))
            images_new.append(image_new)
            masks_new.append(mask_new)
            lbs_weights_new.append(lbs_weight_new)
        images = torch.stack(images_new)
        images = images.reshape(*batch, c, h_scaled, w_scaled)
        masks = torch.stack(masks_new)
        masks = masks.reshape(*batch, h_scaled, w_scaled)
        if lbs_weights_new[0] is not None:
            lbs_weights = torch.stack(lbs_weights_new)
            lbs_weights = lbs_weights.reshape(*batch, h_scaled, w_scaled, d)
        else:
            lbs_weights = None

    if pad:
        return center_pad(images, masks, lbs_weights, intrinsics, shape, bgcolor)
    else:
        return center_crop(images, masks, lbs_weights, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], pad: bool = False, bgcolor: torch.Tensor | None = None) -> AnyViews:
    if "lbs_weights" in views and views["lbs_weights"].shape[-3:-1] != shape:
        lbs_weights = views["lbs_weights"]
    else:
        lbs_weights = None
    images, masks, lbs_weights, intrinsics = rescale_and_crop(
        views["image"], views["mask"], lbs_weights, views["intrinsics"], shape, pad, bgcolor)
    new_views = {
        **views,
        "image": images,
        "mask": masks,
        "intrinsics": intrinsics,
    }
    if lbs_weights is not None:
        new_views["lbs_weights"] = lbs_weights
    return new_views


def apply_crop_shim(example: AnyExample, shape: tuple[int, int], pad: bool = False) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape, pad, example["bgcolor"]),
        "target": apply_crop_shim_to_views(example["target"], shape, pad, example["bgcolor"]),
    }


def random_scale_and_crop(image: torch.Tensor, mask, lbs_weights, intrinsics, bgcolor, scale_range=(0.8, 1.2)) -> torch.Tensor:
    """
    Randomly scale the input image and crop/pad to maintain original size.

    Args:
        image: Input image tensor of shape [H, W, 3]
        scale_range: Range for scaling factor, default (0.8, 1.2)

    Returns:
        Scaled and cropped/padded image tensor of shape [H, W, 3]
    """
    is_numpy = False
    if not torch.is_tensor(image):
        image = torch.from_numpy(image)
        is_numpy = True
    # 获取图像的高度和宽度
    h, w = image.shape[1:]

    # 生成随机缩放因子
    scale_factor = random.uniform(*scale_range)

    # 计算新的高度和宽度
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # 使用 torchvision.transforms.functional.resize 进行缩放
    scaled_image = tvf.resize(image, [new_h, new_w])
    scaled_mask = tvf.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    scaled_lbs_weights = tvf.resize(lbs_weights, [new_h, new_w])
    # intrinsics[..., 0, 0] *= new_w / w
    # intrinsics[..., 1, 1] *= new_h / h

    # 如果缩放后的图像比原图大，进行居中裁剪
    if new_h > h or new_w > w:
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        top = random.randint(0, new_h - h)
        left = random.randint(0, new_w - w)
        scaled_image = scaled_image[:, top:top + h, left:left + w]
        scaled_mask = scaled_mask[:, top:top + h, left:left + w]
        scaled_lbs_weights = scaled_lbs_weights[:, top:top + h, left:left + w]
        intrinsics[..., 0, 0] *= new_w / w
        intrinsics[..., 1, 1] *= new_h / h
        intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] * new_w - left) / w
        intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] * new_h - top) / h
    else:
        # 如果缩放后的图像比原图小，进行居中填充
        # padded_image = torch.ones((3, h, w), dtype=image.dtype)
        padded_image = bgcolor.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
        padded_mask = torch.zeros(1, h, w, dtype=mask.dtype)
        padded_lbs_weights = torch.zeros(55, h, w, dtype=scaled_lbs_weights.dtype)
        # print(padded_image.shape, scaled_image.shape)
        top = h-new_h #(h - new_h) // 2 # H不应该居中
        left = (w - new_w) // 2
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        padded_image[:, top:top + new_h, left:left + new_w] = scaled_image
        scaled_image = padded_image
        padded_mask[:, top:top + new_h, left:left + new_w] = scaled_mask
        scaled_mask = padded_mask
        padded_lbs_weights[:, top:top + new_h, left:left + new_w] = scaled_lbs_weights
        scaled_lbs_weights = padded_lbs_weights
        intrinsics[..., 0, 0] *= new_w / w
        intrinsics[..., 1, 1] *= new_h / h
        intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] * new_w + left) / w
        intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] * new_h + top) / h
    if is_numpy:
        scaled_image = scaled_image.numpy()
    return scaled_image, scaled_mask, scaled_lbs_weights, intrinsics


def apply_crop_shim2(example: AnyExample, shape: tuple[int, int], pad: bool = False) -> AnyExample:
    """This augmentation borrowed from IDOL"""
    cond_imgs = example["context"]["image"]
    cond_masks = example["context"]["mask"][None]
    cond_lbs_weights = example["context"]["lbs_weights"].permute(0, 3, 1, 2)
    cond_imgs[0], cond_masks[0], cond_lbs_weights[0], example["context"]["intrinsics"] = random_scale_and_crop(
        cond_imgs[0], cond_masks[0], cond_lbs_weights[0], example["context"]["intrinsics"], example["bgcolor"],
        (0.7, 1.1))

    example["context"]["image"] = cond_imgs
    example["context"]["mask"] = cond_masks.squeeze(1)
    example["context"]["lbs_weights"] = cond_lbs_weights.permute(0, 2, 3, 1)

    return apply_crop_shim(example, shape, pad)
