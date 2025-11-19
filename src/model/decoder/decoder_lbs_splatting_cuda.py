from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput

from ...misc.body_utils import apply_lbs_to_gaussians
from ...misc.cam_utils import world_to_pix
from .cuda_splatting import Renderer


@dataclass
class DecoderLBSSplattingCUDACfg:
    name: Literal["lbs_splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    render_no_canonical: bool = False
    render_no_rgb: bool = False
    render_context: bool = False
    use_sh: bool = True
    warp_by_shape: bool = False


class DecoderLBSSplattingCUDA(Decoder[DecoderLBSSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderLBSSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

        self.render_cuda = Renderer()

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        Rs: Float[Tensor, "batch view dim 3 3"],
        Ts: Float[Tensor, "batch view dim 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        bgcolor: Float[Tensor, "batch 3"] | None = None,
        context_extrinsics: Float[Tensor, "batch viewc 4 4"] | None = None,
        context_intrinsics: Float[Tensor, "batch viewc 3 3"] | None = None,
        context_Rs: Float[Tensor, "batch viewc dim 3 3"] | None = None,
        context_Ts: Float[Tensor, "batch viewc dim 3"] | None = None,
        context_near: Float[Tensor, "batch viewc"] | None = None,
        context_far: Float[Tensor, "batch viewc"] | None = None,
        cnl_Rs = None, cnl_Ts = None,
        context_cnl_Rs = None, context_cnl_Ts = None,
        return_correspondence: bool = False,
    ) -> tuple[DecoderOutput, dict] | tuple[DecoderOutput, dict, Float[Tensor, "batch view gaussians 2"]]:
        b, v, _, _ = extrinsics.shape

        if self.cfg.warp_by_shape:
            means, covariances = apply_lbs_to_gaussians(
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                rearrange(cnl_Rs, "b v w i j -> (b v) w i j").contiguous(),
                rearrange(cnl_Ts, "b v w k -> (b v) w k").contiguous(),
                repeat(F.softmax(gaussians.lbs_weights_bones, dim=-1), "b g w -> (b v) g w", v=v),
            )
            means, covariances = apply_lbs_to_gaussians(
                means,
                covariances,
                rearrange(Rs, "b v w i j -> (b v) w i j").contiguous(),
                rearrange(Ts, "b v w k -> (b v) w k").contiguous(),
                repeat(F.softmax(gaussians.lbs_weights, dim=-1), "b g w -> (b v) g w", v=v),
            )
        else:
            means, covariances = apply_lbs_to_gaussians(
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                rearrange(Rs, "b v w i j -> (b v) w i j").contiguous(),
                rearrange(Ts, "b v w k -> (b v) w k").contiguous(),
                repeat(F.softmax(gaussians.lbs_weights, dim=-1), "b g w -> (b v) g w", v=v),
            )

        if bgcolor is None:
            bgcolor_target = repeat(self.background_color, "c -> (b v) c", b=b, v=v)
        else:
            bgcolor_target = repeat(bgcolor, "b c -> (b v) c", v=v)
        color, depth, opacity, conf, *infos = self.render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j").contiguous(),
            rearrange(intrinsics, "b v i j -> (b v) i j").contiguous(),
            rearrange(near, "b v -> (b v)").contiguous(),
            rearrange(far, "b v -> (b v)").contiguous(),
            image_shape,
            bgcolor_target,
            means,
            covariances,
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            gaussian_confs=repeat(gaussians.conf, "b g -> (b v) g", v=v) if gaussians.conf is not None else None,
            scale_invariant=self.make_scale_invariant,
            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i").contiguous() if cam_rot_delta is not None else None,
            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i").contiguous() if cam_trans_delta is not None else None,
            use_sh=self.cfg.use_sh,
            return_n_touched=return_correspondence,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v).contiguous()
        depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v).contiguous()
        conf = None if conf is None else rearrange(conf, "(b v) h w -> b v h w", b=b, v=v).contiguous()
        output = DecoderOutput(color, depth, conf)

        # there are a fixed number of gaussians from uv template
        # so the first n_template gaussians are from uv template
        # use this to save memory
        if self.cfg.render_no_rgb or self.cfg.render_no_canonical or self.cfg.render_context:
            n_template = torch.nonzero(gaussians.idx[..., 0] == 0)[:, 1].max().item() + 1

        opacities = gaussians.opacities
        output_aux = {}
        if self.cfg.render_no_rgb:
            color_aux, depth_aux, opacity_aux, conf_aux, *info = self.render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j").contiguous(),
                rearrange(intrinsics, "b v i j -> (b v) i j").contiguous(),
                rearrange(near, "b v -> (b v)").contiguous(),
                rearrange(far, "b v -> (b v)").contiguous(),
                image_shape,
                bgcolor_target,
                means[:, :n_template],
                covariances[:, :n_template],
                repeat(gaussians.harmonics[:, :n_template], "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(opacities[:, :n_template], "b g -> (b v) g", v=v),
                gaussian_confs=repeat(gaussians.conf[:, :n_template], "b g -> (b v) g", v=v) if gaussians.conf is not None else None,
                scale_invariant=self.make_scale_invariant,
                cam_rot_delta=rearrange(cam_rot_delta,
                                        "b v i -> (b v) i").contiguous() if cam_rot_delta is not None else None,
                cam_trans_delta=rearrange(cam_trans_delta,
                                          "b v i -> (b v) i").contiguous() if cam_trans_delta is not None else None,
                use_sh=self.cfg.use_sh,
                return_n_touched=return_correspondence,
            )
            color_aux = rearrange(color_aux, "(b v) c h w -> b v c h w", b=b, v=v).contiguous()
            depth_aux = rearrange(depth_aux, "(b v) h w -> b v h w", b=b, v=v).contiguous()
            conf_aux = None if conf_aux is None else rearrange(conf_aux, "(b v) h w -> b v h w", b=b, v=v).contiguous()

            output_template = DecoderOutput(color_aux, depth_aux, conf_aux)
            output_aux["output_template"] = output_template

        if self.cfg.render_no_canonical:
            color_aux, depth_aux, opacity_aux, conf_aux = self.render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j").contiguous(),
                rearrange(intrinsics, "b v i j -> (b v) i j").contiguous(),
                rearrange(near, "b v -> (b v)").contiguous(),
                rearrange(far, "b v -> (b v)").contiguous(),
                image_shape,
                bgcolor_target,
                means[:, n_template:],
                covariances[:, n_template:],
                repeat(gaussians.harmonics[:, n_template:], "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(opacities[:, n_template:], "b g -> (b v) g", v=v),
                gaussian_confs=repeat(gaussians.conf[:, n_template:], "b g -> (b v) g", v=v) if gaussians.conf is not None else None,
                scale_invariant=self.make_scale_invariant,
                cam_rot_delta=rearrange(cam_rot_delta,
                                        "b v i -> (b v) i").contiguous() if cam_rot_delta is not None else None,
                cam_trans_delta=rearrange(cam_trans_delta,
                                          "b v i -> (b v) i").contiguous() if cam_trans_delta is not None else None,
                use_sh=self.cfg.use_sh,
            )
            color_aux = rearrange(color_aux, "(b v) c h w -> b v c h w", b=b, v=v).contiguous()
            depth_aux = rearrange(depth_aux, "(b v) h w -> b v h w", b=b, v=v).contiguous()
            conf_aux = None if conf_aux is None else rearrange(conf_aux, "(b v) h w -> b v h w", b=b, v=v).contiguous()

            output_img = DecoderOutput(color_aux, depth_aux, conf_aux)
            output_aux["output_img"] = output_img

        if self.cfg.render_context and context_extrinsics is not None:
            _, v_context, *_ = context_extrinsics.shape

            if bgcolor is None:
                bgcolor_context = repeat(self.background_color, "c -> (b v) c", b=b, v=v_context)
            else:
                bgcolor_context = repeat(bgcolor, "b c -> (b v) c", v=v_context)

            if self.cfg.warp_by_shape:
                means, covariances = apply_lbs_to_gaussians(
                    repeat(gaussians.means[:, n_template:], "b g xyz -> (b v) g xyz", v=v_context),
                    repeat(gaussians.covariances[:, n_template:], "b g i j -> (b v) g i j", v=v_context),
                    rearrange(context_cnl_Rs, "b v w i j -> (b v) w i j").contiguous(),
                    rearrange(context_cnl_Ts, "b v w k -> (b v) w k").contiguous(),
                    repeat(F.softmax(gaussians.lbs_weights_bones[:, n_template:], dim=-1), "b g w -> (b v) g w", v=v_context),
                )
                means_context, covariances_context = apply_lbs_to_gaussians(
                    means,
                    covariances,
                    rearrange(context_Rs, "b v w i j -> (b v) w i j").contiguous(),
                    rearrange(context_Ts, "b v w k -> (b v) w k").contiguous(),
                    repeat(F.softmax(gaussians.lbs_weights[:, n_template:], dim=-1), "b g w -> (b v) g w", v=v_context),
                )
            else:
                means_context, covariances_context = apply_lbs_to_gaussians(
                    repeat(gaussians.means[:, n_template:], "b g xyz -> (b v) g xyz", v=v_context),
                    repeat(gaussians.covariances[:, n_template:], "b g i j -> (b v) g i j", v=v_context),
                    rearrange(context_Rs, "b v w i j -> (b v) w i j").contiguous(),
                    rearrange(context_Ts, "b v w k -> (b v) w k").contiguous(),
                    repeat(F.softmax(gaussians.lbs_weights[:, n_template:], dim=-1), "b g w -> (b v) g w", v=v_context),
                )

            color_aux, depth_aux, opacity_aux, conf_aux = self.render_cuda(
                rearrange(context_extrinsics, "b v i j -> (b v) i j").contiguous(),
                rearrange(context_intrinsics, "b v i j -> (b v) i j").contiguous(),
                rearrange(context_near, "b v -> (b v)").contiguous(),
                rearrange(context_far, "b v -> (b v)").contiguous(),
                image_shape,
                bgcolor_context,
                means_context,
                covariances_context,
                repeat(gaussians.harmonics[:, n_template:], "b g c d_sh -> (b v) g c d_sh", v=v_context),
                repeat(opacities[:, n_template:], "b g -> (b v) g", v=v_context),
                gaussian_confs=repeat(gaussians.conf[:, n_template:], "b g -> (b v) g", v=v_context) if gaussians.conf is not None else None,
                scale_invariant=self.make_scale_invariant,
                use_sh=self.cfg.use_sh,
            )
            color_aux = rearrange(color_aux, "(b v) c h w -> b v c h w", b=b, v=v_context).contiguous()
            depth_aux = rearrange(depth_aux, "(b v) h w -> b v h w", b=b, v=v_context).contiguous()
            conf_aux = None if conf_aux is None else rearrange(conf_aux, "(b v) h w -> b v h w", b=b, v=v_context).contiguous()

            output_context = DecoderOutput(color_aux, depth_aux, conf_aux)
            output_aux["output_context"] = output_context
        return output, output_aux

