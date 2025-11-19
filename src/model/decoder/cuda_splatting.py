from math import isqrt
from typing import Literal, Optional

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor
import torch.nn as nn

from ...geometry.projection import get_fov, homogenize_points


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    intrinsics: Float[Tensor, " batch 3 3"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)

    near_fx = near / intrinsics[:, 0, 0]
    near_fy = near / intrinsics[:, 1, 1]
    left = - (1 - intrinsics[:, 0, 2]) * near_fx
    right = intrinsics[:, 0, 2] * near_fx
    bottom = (intrinsics[:, 1, 2] - 1) * near_fy
    top = intrinsics[:, 1, 2] * near_fy

    z_sign = 1.0
    result[:, 0, 0] = 2.0 * near / (right - left)
    result[:, 1, 1] = 2.0 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = z_sign
    result[:, 2, 2] = z_sign * far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)

    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_confs: Float[Tensor, "batch gaussian"] | None = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
    cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
    cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
    return_n_touched: bool = False,
) -> (tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"], Float[Tensor, "batch 1 height width"], Float[Tensor, "batch height width"] | None]
      | tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"], Float[Tensor, "batch 1 height width"],  Float[Tensor, "batch height width"] | None, Optional[Int[Tensor, "batch gaussian"]]]):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, intrinsics)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i").contiguous()
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i").contiguous()
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    all_depths = []
    all_opacities = []
    all_n_touched = []
    all_confs = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
            theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
            rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
        )

        all_images.append(image)
        all_radii.append(radii)
        all_depths.append(depth.squeeze(0))
        all_opacities.append(opacity)
        all_n_touched.append(n_touched)

        if gaussian_confs is not None:
            settings = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=tan_fov_x[i].item(),
                tanfovy=tan_fov_y[i].item(),
                bg=torch.ones_like(background_color[i]),
                scale_modifier=1.0,
                viewmatrix=view_matrix[i],
                projmatrix=full_projection[i],
                projmatrix_raw=projection_matrix[i],
                sh_degree=degree,
                campos=extrinsics[i, :3, 3],
                prefiltered=False,  # This matches the original usage.
                debug=False,
            )
            rasterizer = GaussianRasterizer(settings)

            conf, _, _, _, _ = rasterizer(
                means3D=gaussian_means[i],
                means2D=mean_gradients,
                colors_precomp=repeat(gaussian_confs[i], "g -> g c", c=3),
                opacities=gaussian_opacities[i, ..., None],
                cov3D_precomp=gaussian_covariances[i, :, row, col],
                theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
                rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
            )
            conf = conf[0]
            all_confs.append(conf)

    if gaussian_confs is not None:
        all_confs = torch.stack(all_confs)
    else:
        all_confs = None

    if return_n_touched:
        return torch.stack(all_images), torch.stack(all_depths), torch.stack(all_opacities), all_confs, torch.stack(all_n_touched)
    return torch.stack(all_images), torch.stack(all_depths), torch.stack(all_opacities), all_confs


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, " batch"],
        far: Float[Tensor, " batch"],
        image_shape: tuple[int, int],
        background_color: Float[Tensor, "batch 3"],
        gaussian_means: Float[Tensor, "batch gaussian 3"],
        gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
        gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
        gaussian_opacities: Float[Tensor, "batch gaussian"],
        gaussian_confs: Float[Tensor, "batch gaussian"] | None = None,
        scale_invariant: bool = True,
        use_sh: bool = True,
        cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
        return_n_touched: bool = False,
    ) -> (tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"], Float[Tensor, "batch 1 height width"], Float[Tensor, "batch height width"] | None]
      | tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"], Float[Tensor, "batch 1 height width"],  Float[Tensor, "batch height width"] | None, Optional[Int[Tensor, "batch gaussian"]]]):
        return render_cuda(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            background_color,
            gaussian_means,
            gaussian_covariances,
            gaussian_sh_coefficients,
            gaussian_opacities,
            gaussian_confs,
            scale_invariant,
            use_sh,
            cam_rot_delta,
            cam_trans_delta,
            return_n_touched=return_n_touched,
        )



def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    intrinsics = torch.zeros([3, 3], device=extrinsics.device, dtype=torch.float32)
    intrinsics[0, 0] = 0.5 / tan_fov_x
    intrinsics[1, 1] = 0.5 / tan_fov_y
    intrinsics[0, 2] = 0.5
    intrinsics[1, 2] = 0.5
    intrinsics[2, 2] = 1
    projection_matrix = get_projection_matrix(
        near, far, repeat(intrinsics, "i j-> b i j", b=b)
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i").contiguous()
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i").contiguous()
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]
