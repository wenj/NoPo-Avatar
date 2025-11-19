import os.path
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_lbs_adapter import GaussianLBSAdapter, GaussianLBSAdapterCfg, UnifiedGaussianLBSAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.utils import inverse_normalize
from ...misc.body_utils import SMPLX_N_BONES, SMPL_N_BONES, bone_lbs_weights_to_joint_lbs_weights


inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderLBSNoPoSplatCfg:
    name: Literal["template_uv_concat_bone"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianLBSAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    pts3d_head_type: str
    pts3d_head_skip: bool
    gs_params_head_type: str
    input_mean: list[float] = (0.5, 0.5, 0.5)
    input_std: list[float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pretrained_template_reinit: bool = False
    pose_free: bool = True
    apply_mask: str = "none"
    pts3d_for_lbs_weights: bool = False
    highres_uv: bool = False

    has_conf: bool | None = None

    separate_xyz_head: bool = False
    n_hooks: int = 4

    debug: bool = False


class EncoderTemplateUVConcatBone(Encoder[EncoderLBSNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianLBSAdapter

    def __init__(self, cfg: EncoderLBSNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianLBSAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianLBSAdapter(cfg.gaussian_adapter)
        self.apply_mask = cfg.apply_mask

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in - 55 + SMPLX_N_BONES  # 1 for opacity

        self.pretrained_template_reinit = cfg.pretrained_template_reinit

        self.pts3d_head_type = cfg.pts3d_head_type
        self.gs_params_head_type = cfg.gs_params_head_type

        self.separate_xyz_head = cfg.separate_xyz_head
        self.n_hooks = cfg.n_hooks
        self.set_mean_head(output_mode='pts3d', head_type=cfg.pts3d_head_type, landscape_only=False,
                           depth_mode=('exp', -inf, inf), conf_mode=("exp", 1, inf) if self.cfg.has_conf else None,
                           skip=cfg.pts3d_head_skip)
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        pts3d_mean = torch.tensor([0., 0., 0.], dtype=torch.float32)
        pts3d_std = torch.tensor([1., 1., 1.], dtype=torch.float32)
        self.register_buffer('pts3d_mean', pts3d_mean)
        self.register_buffer('pts3d_std', pts3d_std)

        self.debug = cfg.debug

    def set_mean_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, skip):
        # self.output_mode = output_mode
        # self.head_type = head_type
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        if self.pts3d_head_type == 'dpt':
            self.downstream_head1_template = head_factory(head_type, output_mode, self.backbone,
                                                          has_conf=bool(conf_mode),
                                                          out_nchan=3 + 3 * self.separate_xyz_head,
                                                          n_hooks=self.n_hooks)
            self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode),
                                                 skip=skip, out_nchan=3 + 3 * self.separate_xyz_head)

            # magic wrapper
            self.head1_template = transpose_to_landscape(self.downstream_head1_template, activate=landscape_only)
            self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        else:
            self.downstream_head1_template = head_factory(head_type, output_mode, self.backbone,
                                                          has_conf=bool(conf_mode), img_nchan=55 + 3,
                                                          n_hooks=self.n_hooks)
            self.downstream_head2_rgb = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

            # magic wrapper
            self.head1_template = transpose_to_landscape(self.downstream_head1_template, activate=landscape_only)
            self.head2 = transpose_to_landscape(self.downstream_head2_rgb, activate=landscape_only)

        if self.pretrained_template_reinit:
            nn.init.uniform_(self.downstream_head1_template.dpt.head[-1].weight, -1e-5, 1e-5)
            nn.init.zeros_(self.downstream_head1_template.dpt.head[-1].bias)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                    out_nchan=self.raw_gs_dim)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                     out_nchan=self.raw_gs_dim)  # for view2 3DGS

            # # magic wrapper
            # self.head3 = transpose_to_landscape(self.to_gaussians, activate=landscape_only)
            # self.head4 = transpose_to_landscape(self.to_gaussians2, activate=landscape_only)
        elif head_type == 'dpt_gs' or head_type == 'dpt_gs_debug':
            self.gaussian_param_head_template = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                             out_nchan=self.raw_gs_dim + SMPL_N_BONES, img_nchan=55 + 3,
                                                             n_hooks=self.n_hooks)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                     out_nchan=self.raw_gs_dim + SMPL_N_BONES)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
            self,
            pdf: Float[Tensor, " *batch"],
            global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2 ** x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        if head_num == 1:
            head = self.head1_template
        else:
            head = self.head2
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def filter_by_mask(self, gaussians_template, masks_template, gaussians, masks):

        b, v, r, srf, spp, xyz = gaussians.means.shape

        masks_template = repeat(masks_template, "b v r srf c -> b v r srf spp c", spp=spp)
        masks_template = rearrange(masks_template.squeeze(-1), "b v r srf spp -> b (v r srf spp)").contiguous()

        masks = repeat(masks, "b v r srf c -> b v r srf spp c", spp=spp)
        masks = rearrange(masks.squeeze(-1), "b v r srf spp -> b (v r srf spp)").contiguous()

        if self.debug:
            means_ori = torch.cat([
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ).contiguous()
            ], dim=1)
            covariance_ori = torch.cat([
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ).contiguous()
            ], dim=1)
            harmonics_ori = torch.cat([
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ).contiguous(),
            ], dim=1)
            opacities_ori = torch.cat([
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ).contiguous(),
            ], dim=1)
            lbs_weights_ori = torch.cat([
                rearrange(
                    gaussians.lbs_weights,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous()
            ], dim=1)
            idx_ori = torch.cat([
                rearrange(
                    gaussians.idx,
                    "b v r srf spp c -> b (v r srf spp) c",
                ).contiguous()
            ], dim=1)
            if gaussians.conf is not None:
                conf_ori = torch.cat([
                    rearrange(
                        gaussians.conf,
                        "b v r srf spp -> b (v r srf spp)",
                    ).contiguous()
                ], dim=1)
                conf = []
        else:
            masks = torch.cat([masks_template, masks], dim=1)
            means_ori = torch.cat([
                rearrange(
                    gaussians_template.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ).contiguous(),
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ).contiguous()
            ], dim=1)
            covariance_ori = torch.cat([
                rearrange(
                    gaussians_template.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ).contiguous(),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ).contiguous()
            ], dim=1)
            harmonics_ori = torch.cat([
                rearrange(
                    gaussians_template.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ).contiguous(),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ).contiguous(),
            ], dim=1)
            opacities_ori = torch.cat([
                rearrange(
                    gaussians_template.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ).contiguous(),
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ).contiguous(),
            ], dim=1)
            lbs_weights_ori = torch.cat([
                rearrange(
                    gaussians_template.lbs_weights,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous(),
                rearrange(
                    gaussians.lbs_weights,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous()
            ], dim=1)
            lbs_weights_bone_ori = torch.cat([
                rearrange(
                    gaussians_template.lbs_weights_bones,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous(),
                rearrange(
                    gaussians.lbs_weights_bones,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous()
            ], dim=1)
            idx_ori = torch.cat([
                rearrange(
                    gaussians_template.idx,
                    "b v r srf spp c -> b (v r srf spp) c",
                ).contiguous(),
                rearrange(
                    gaussians.idx,
                    "b v r srf spp c -> b (v r srf spp) c",
                ).contiguous()
            ], dim=1)
            if gaussians.conf is not None:
                conf_ori = torch.cat([
                    rearrange(
                        gaussians_template.conf,
                        "b v r srf spp -> b (v r srf spp)",
                    ).contiguous(),
                    rearrange(
                        gaussians.conf,
                        "b v r srf spp -> b (v r srf spp)",
                    ).contiguous()
                ], dim=1)
                conf = []

        means, covariances, harmonics, opacities, lbs_weights, lbs_weights_bone, idx = [], [], [], [], [], [], []
        nums = []
        for bi in range(b):
            valid = masks[bi] > 1e-3
            means.append(means_ori[bi][valid])
            covariances.append(covariance_ori[bi][valid])
            harmonics.append(harmonics_ori[bi][valid])
            opacities.append(opacities_ori[bi][valid])
            lbs_weights.append(lbs_weights_ori[bi][valid])
            lbs_weights_bone.append(lbs_weights_bone_ori[bi][valid])
            idx.append(idx_ori[bi][valid])
            nums.append(means[-1].shape[0])
            if gaussians.conf is not None:
                conf.append(conf_ori[bi][valid])

        max_num = max(nums)
        means_new = means_ori.new_zeros(b, max_num, *means_ori.shape[2:])
        means_new.fill_(1e8)
        covariances_new = covariance_ori.new_zeros(b, max_num, *covariance_ori.shape[2:])
        harmonics_new = harmonics_ori.new_zeros(b, max_num, *harmonics_ori.shape[2:])
        opacities_new = opacities_ori.new_zeros(b, max_num, *opacities_ori.shape[2:])
        lbs_weights_new = lbs_weights_ori.new_zeros(b, max_num, *lbs_weights_ori.shape[2:])
        lbs_weights_bone_new = lbs_weights_bone_ori.new_zeros(b, max_num, *lbs_weights_bone_ori.shape[2:])
        idx_new = idx_ori.new_full((b, max_num, *idx_ori.shape[2:]), -1)
        if gaussians.conf is not None:
            conf_new = conf_ori.new_ones(b, max_num, *conf_ori.shape[2:])
        else:
            conf_new = None
        for bi in range(b):
            means_new[bi][:nums[bi]] = means[bi]
            covariances_new[bi][:nums[bi]] = covariances[bi]
            harmonics_new[bi][:nums[bi]] = harmonics[bi]
            opacities_new[bi][:nums[bi]] = opacities[bi]
            lbs_weights_new[bi][:nums[bi]] = lbs_weights[bi]
            lbs_weights_bone_new[bi][:nums[bi]] = lbs_weights_bone[bi]
            idx_new[bi][:nums[bi]] = idx[bi]
            if gaussians.conf is not None:
                conf_new[bi][:nums[bi]] = conf[bi]

        return Gaussians(means_new, covariances_new, harmonics_new, opacities_new, lbs_weights_new, idx=idx_new,
                         nums=nums, conf=conf_new, lbs_weights_bones=lbs_weights_bone_new, )

    def forward(
            self,
            context: dict,
            global_step: int = 0,
            visualization_dump: Optional[dict] = None,
            return_complete_gaussians_rgb: bool = False,
            return_complete_gaussians: bool = False,
    ):
        use_smplx = context["use_smplx"].any()
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        rgbs = rearrange(inverse_normalize(context["image"]), "b v c h w -> b v h w c").contiguous()
        rgbs = repeat(rgbs, "b v h w c -> b v h w srf c", srf=self.cfg.num_surfaces)
        rgbs = rearrange(rgbs, "b v h w srf c -> b v (h w) srf () c").contiguous()

        # Encode the context images.
        [dec_feat_template, dec_feat], [shape_template, shape], [template, images] = self.backbone(context)
        with torch.amp.autocast('cuda', enabled=False):
            if self.pts3d_head_type == 'dpt':
                res1, _ = self._downstream_head(1, [tok.float() for tok in dec_feat_template] + [template],
                                                shape_template)
                if use_smplx or not self.separate_xyz_head:
                    res1['pts3d'] = res1['pts3d'][..., :3] + context["template_3d"]
                else:
                    res1['pts3d'] = res1['pts3d'][..., 3:] + context["template_3d"]
                all_mean_res = []
                for i in range(v):
                    res2, _ = self._downstream_head(2, [tok[:, i].float() for tok in dec_feat] + [images[:, i]],
                                                    shape[:, i])
                    if use_smplx or not self.separate_xyz_head:
                        res2['pts3d'] = res2['pts3d'][..., :3]
                    else:
                        res2['pts3d'] = res2['pts3d'][..., 3:]
                    all_mean_res.append(res2)
            else:
                res1 = self.downstream_head1_template([tok.float() for tok in dec_feat_template], None, template,
                                                      shape_template[0].cpu().tolist())
                res1['pts3d'] += context["template_3d"]
                all_mean_res = []
                for i in range(v):
                    res2 = self.downstream_head2_rgb([tok[:, i].float() for tok in dec_feat], None, images[:, i, :3],
                                                     shape[0, i].cpu().tolist())
                    all_mean_res.append(res2)

            # for the 3DGS heads
            if self.gs_params_head_type == 'dpt_gs' or self.gs_params_head_type == 'dpt_gs_debug':
                GS_res1 = self.gaussian_param_head_template([tok.float() for tok in dec_feat_template],
                                                            all_mean_res[0]['pts3d'].permute(0, 3, 1, 2), template,
                                                            shape_template[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d").contiguous()
                all_other_params = []
                for i in range(v):
                    GS_res2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat],
                                                        all_mean_res[i]['pts3d'].permute(0, 3, 1, 2), images[:, i, :3],
                                                        shape[0, i].cpu().tolist())
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d").contiguous()
                    all_other_params.append(GS_res2)
            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

        # first wrap up the prediction for template branch
        pts_template = res1['pts3d']
        pts_template = rearrange(pts_template, "b h w xyz -> b () (h w) xyz").contiguous()
        pts_template = pts_template.unsqueeze(-2)  # for cfg.num_surfaces

        depths_template = pts_template[..., -1].unsqueeze(-1)

        gaussians_template = GS_res1.unsqueeze(1)
        gaussians_template = rearrange(gaussians_template, "... (srf c) -> ... srf c",
                                       srf=self.cfg.num_surfaces).contiguous()
        densities_template = gaussians_template[..., 0].sigmoid().unsqueeze(-1)

        if use_smplx:
            lbs_weights_bones_template = gaussians_template[..., -(SMPL_N_BONES + SMPLX_N_BONES):-SMPL_N_BONES]
        else:
            lbs_weights_bones_template = gaussians_template[..., -SMPL_N_BONES:]
        # print('bone', lbs_weights_bones_template.isnan().any(), lbs_weights_bones_template.isinf().any())
        lbs_weights_joints_template = bone_lbs_weights_to_joint_lbs_weights(lbs_weights_bones_template,
                                                                            use_smplx=use_smplx)
        # print('joint', lbs_weights_joints_template.isnan().any(), lbs_weights_joints_template.isinf().any())
        gaussians_template = torch.concatenate(
            [gaussians_template[..., :-(SMPL_N_BONES + SMPLX_N_BONES)], lbs_weights_joints_template], dim=-1)

        opacities_template = self.map_pdf_to_opacity(densities_template, global_step)
        # insert template uv as canonical view
        mask_template = repeat(context["template_mask"], "b h w -> b () h w srf c", srf=self.cfg.num_surfaces, c=1)
        mask_template = rearrange(mask_template, "b v h w srf c -> b v (h w) srf c").contiguous()
        if self.apply_mask != 'none':
            if self.apply_mask == 'soft':
                opacities_template *= mask_template
            else:
                opacities_template = mask_template

        # Convert the features and depths into Gaussians.
        gaussians_template = self.gaussian_adapter.forward(
            pts_template.unsqueeze(-2),
            depths_template,
            opacities_template,
            rearrange(gaussians_template[..., 1:], "b v r srf c -> b v r srf () c").contiguous(),
            use_smplx=use_smplx
        )
        h_template, w_template = context["template_mask"].shape[-2:]
        idx_template = torch.stack(
            torch.meshgrid(torch.arange(1, device=device), torch.arange(h_template, device=device),
                           torch.arange(w_template, device=device)), dim=-1)
        idx_template = repeat(idx_template, "v h w c -> b v h w srf spp c", b=b, srf=self.cfg.num_surfaces,
                              spp=1).contiguous()
        gaussians_template.idx = rearrange(idx_template,
                                           "b v h w srf spp c -> b v (h w) srf spp c").contiguous()  # view_id, h, w
        gaussians_template.lbs_weights_bones = rearrange(lbs_weights_bones_template,
                                                         "b v h w c -> b v (h w) () () c").contiguous()

        if 'conf' in res1:
            conf_template = res1['conf']
            conf_template = repeat(conf_template, "b h w -> b v (h w) srf spp", v=1, srf=self.cfg.num_surfaces, spp=1)
            gaussians_template.conf = conf_template
        else:
            gaussians_template.conf = None

        # now handle rgb branches
        pts_all = [all_mean_res_i['pts3d'] for all_mean_res_i in all_mean_res]
        pts_all = torch.stack(pts_all, dim=1)
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz").contiguous()
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = torch.stack(all_other_params, dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces).contiguous()
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        if use_smplx:
            lbs_weights_bones = gaussians[..., -(SMPL_N_BONES + SMPLX_N_BONES):-SMPL_N_BONES]
        else:
            lbs_weights_bones = gaussians[..., -SMPL_N_BONES:]
        # print('bone', lbs_weights_bones.isnan().any(), lbs_weights_bones.isinf().any())
        lbs_weights_joints = bone_lbs_weights_to_joint_lbs_weights(lbs_weights_bones, use_smplx=use_smplx)
        # print('joint', lbs_weights_joints.isnan().any(), lbs_weights_joints.isinf().any())
        gaussians = torch.concatenate(
            [gaussians[..., :-(SMPL_N_BONES + SMPLX_N_BONES)], lbs_weights_joints], dim=-1)

        opacities = self.map_pdf_to_opacity(densities, global_step)
        # insert template uv as canonical view
        mask = repeat(context["mask"], "b v h w -> b v h w srf c", srf=self.cfg.num_surfaces, c=1)
        mask = rearrange(mask, "b v h w srf c -> b v (h w) srf c").contiguous()
        if self.apply_mask != 'none':
            if self.apply_mask == 'soft':
                opacities *= mask
            else:
                opacities = mask

        # Convert the features and depths into Gaussians.
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            opacities,
            rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c").contiguous(),
            use_smplx=use_smplx
        )
        idx = torch.stack(torch.meshgrid(torch.arange(1, v + 1, device=device), torch.arange(h, device=device),
                                         torch.arange(w, device=device)), dim=-1)
        idx = repeat(idx, "v h w c -> b v h w srf spp c", b=b, srf=self.cfg.num_surfaces, spp=1).contiguous()
        gaussians.idx = rearrange(idx, "b v h w srf spp c -> b v (h w) srf spp c").contiguous()  # view_id, h, w
        gaussians.lbs_weights_bones = rearrange(lbs_weights_bones, "b v h w c -> b v (h w) () () c")

        if "conf" in all_mean_res[0].keys():
            conf_all = [all_mean_res_i['conf'] for all_mean_res_i in all_mean_res]
            conf_all = torch.stack(conf_all, dim=1)
            conf_all = repeat(conf_all, "b v h w -> b v (h w) srf spp", srf=self.cfg.num_surfaces, spp=1).contiguous()
            gaussians.conf = conf_all
        else:
            gaussians.conf = None

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).contiguous()
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            ).contiguous()
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            ).contiguous()
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            ).contiguous()
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).contiguous()

        if return_complete_gaussians_rgb:
            if "lbs_weights" in context:
                context["lbs_weights"] = torch.cat(
                    [context["template_lbs_weights"].unsqueeze(1), context["lbs_weights"]], dim=1)
                context["mask"] = torch.cat([context["template_mask"].unsqueeze(1), context["mask"]], dim=1)
            return self.filter_by_mask(gaussians_template, mask_template, gaussians, mask), Gaussians(
                rearrange(
                    torch.cat([gaussians_template.means, gaussians.means], dim=1),
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    torch.cat([gaussians_template.covariances, gaussians.covariances], dim=1),
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    torch.cat([gaussians_template.harmonics, gaussians.harmonics], dim=1),
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    torch.cat([gaussians_template.opacities, gaussians.opacities], dim=1),
                    "b v r srf spp -> b (v r srf spp)",
                ),
                rearrange(
                    torch.cat([gaussians_template.lbs_weights, gaussians.lbs_weights], dim=1),
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                lbs_weights_bones=rearrange(
                    torch.cat([gaussians_template.lbs_weights_bones, gaussians.lbs_weights_bones], dim=1),
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
            )

        if return_complete_gaussians:
            gall = Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                rearrange(
                    gaussians.lbs_weights,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                lbs_weights_bones=rearrange(
                    gaussians.lbs_weights_bones,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                idx=rearrange(
                    gaussians.idx,
                    "b v r srf spp d -> b (v r srf spp) d"
                )
            )
            gall_template = Gaussians(
                rearrange(
                    gaussians_template.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians_template.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians_template.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    gaussians_template.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                rearrange(
                    gaussians_template.lbs_weights,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                lbs_weights_bones=rearrange(
                    gaussians_template.lbs_weights_bones,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
            )
            return self.filter_by_mask(gaussians_template, mask_template, gaussians, mask), gall_template, gall

        return self.filter_by_mask(gaussians_template, mask_template, gaussians, mask)

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim