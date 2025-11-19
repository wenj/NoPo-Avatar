from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any, Literal

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only, grad_norm
from tabulate import tabulate
from torch import Tensor, nn, optim
import numpy as np
from PIL import Image
import copy
import seaborn as sns
import cv2
import os
import json
import torch.utils.checkpoint as checkpoint

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose, get_camrot, rotate_camera_by_frame_idx
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .encoder.common.gaussians import build_covariance
from .types import Gaussians

from ..misc.body_utils import apply_lbs_to_means, get_canonical_global_tfms, body_pose_to_body_RTs, get_global_RTs, apply_global_tfm_to_camera, \
    get_canonical_global_tfms_tensor, body_pose_to_body_RTs_tensor, get_global_RTs_tensor, get_canonical_tfms, _rvec_to_rmtx
from ..misc.cam_utils import get_camrot

import pytorch3d


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    align_human_pose: Literal["disabled", "context", "target"]
    human_rot_opt_lr: float
    human_trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    save_canonical: bool
    save_gaussians: bool
    save_lbs_weights: bool

    refine_gaussians: bool = False
    gaussian_scale_opt_lr: float = 1e-3
    gaussian_rot_opt_lr: float = 1e-3
    refine_gaussians_steps: int = 10
    pose_seq: str | None = None
    optim_pose: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    random_n_views: bool = False


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

        from smplx import SMPLX
        import os
        MODEL_DIR = "datasets/smplx"
        self.smplx_model = SMPLX(
            model_path=os.path.join(MODEL_DIR, 'SMPLX_NEUTRAL.npz'),
            use_pca=False,
            num_pca_comps=12,
            num_betas=10,
            flat_hand_mean=True)

        smplx = SMPLX(model_path=os.path.join(MODEL_DIR, 'SMPLX_MALE.npz'))
        self.template_tpose_joints = smplx().joints.detach().cpu()[0, :55]
        self.pose_mean = smplx.pose_mean.cuda()

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch = batch[torch.randint(len(batch), size=(1,)).item()]
        batch: BatchedExample = self.data_shim(batch)
        if self.train_cfg.random_n_views:
            N = batch["context"]["image"].shape[1]
            n_views = np.random.randint(1, N + 1)
            new_context = {}
            for key in batch["context"].keys():
                if key not in ["overlap", "use_smplx"] and not key.startswith("template"):
                    new_context[key] = batch["context"][key][:, :n_views]
                else:
                    new_context[key] = batch["context"][key]
            batch["context"] = new_context
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        visualization_dump = {}
        if self.distiller is not None:
            visualization_dump = {}
        gaussians, gaussians_rgb = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, return_complete_gaussians_rgb=True)

        torch.cuda.empty_cache()

        output, output_aux = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["Rs"],
            batch["target"]["Ts"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
            bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
            context_extrinsics=batch["context"]["extrinsics"],
            context_intrinsics=batch["context"]["intrinsics"],
            context_Rs=batch["context"]["Rs"],
            context_Ts=batch["context"]["Ts"],
            context_near=batch["context"]["near"],
            context_far=batch["context"]["far"],
            cnl_Rs=batch["target"]["cnl_Rs"],
            cnl_Ts=batch["target"]["cnl_Ts"],
            context_cnl_Rs=batch["context"]["cnl_Rs"],
            context_cnl_Ts=batch["context"]["cnl_Ts"],
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # unprepare the image for rendering
        batch["context"]["image"] = inverse_normalize(batch["context"]["image"], self.encoder.cfg.input_mean, self.encoder.cfg.input_std)

        # Compute and log loss.
        total_loss = 0
        loss_dict = {}
        for loss_fn in self.losses:
            if loss_fn.name in ["lbs_weights", "pts3d"]:
                loss = loss_fn.forward(output, batch, gaussians_rgb, self.global_step)
            elif loss_fn.name == "noise":
                loss = loss_fn.forward(output_aux["output_nosie"], batch, gaussians, self.global_step)
            else:
                loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            loss_dict[loss_fn.name] = loss

            loss_aux = 0.
            if "output_img" in output_aux and loss_fn.name in ["lpips", "mse", "ssim"]:
                loss_img = loss_fn.forward(output_aux["output_img"], batch, gaussians, self.global_step, weight="rgb")
                self.log(f"loss/{loss_fn.name}_img", loss_img)
                loss_aux += loss_img
                loss_dict[loss_fn.name + "_img"] = loss_img

            if "output_template" in output_aux and loss_fn.name in ["lpips", "mse", "ssim"]:
                loss_template = loss_fn.forward(output_aux["output_template"], batch, gaussians, self.global_step, weight="template")
                self.log(f"loss/{loss_fn.name}_template", loss_template)
                loss_aux += loss_template
                loss_dict[loss_fn.name + "_template"] = loss_template

            if "output_context" in output_aux and loss_fn.name in ["lpips", "mse", "ssim"]:
                loss_context = loss_fn.forward(output_aux["output_context"], batch, gaussians, self.global_step, compare_target=False, weight="rgb")
                self.log(f"loss/{loss_fn.name}_context", loss_context)
                loss_aux += loss_context
                loss_dict[loss_fn.name + "_context"] = loss_context

            total_loss = total_loss + loss + loss_aux

        # distillation
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            distillation_loss = self.distiller_loss(pseudo_gt1['pts3d'], pseudo_gt2['pts3d'],
                                                    visualization_dump['means'][:, 0].squeeze(-2),
                                                    visualization_dump['means'][:, 1].squeeze(-2),
                                                    pseudo_gt1['conf'], pseudo_gt2['conf'], disable_view1=False) * 0.1
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            loss_str = ""
            for key, value in loss_dict.items():
                loss_str += f"{key} = {value:.6f}, "
            loss_str = loss_str[:-2]
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; target = {batch['target']['index'].tolist()}; "
                f"loss = {total_loss:.6f} ({loss_str})", flush=True
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        if "opacities" in visualization_dump and (self.global_rank == 0 and self.global_step % self.train_cfg.log_every_n_steps == 0):
            context_img = batch["context"]["image"][0]
            rgb_gt = batch["target"]["image"][0]
            rgb_pred = output.color[0]
            if "output_img" in output_aux:
                rgb_pred_img = output_aux["output_img"].color[0]
            else:
                rgb_pred_img = None

            if "output_template" in output_aux:
                rgb_pred_template = output_aux["output_template"].color[0]
            else:
                rgb_pred_template = None

            if "output_context" in output_aux:
                rgb_pred_context = output_aux["output_context"].color[0]
            else:
                rgb_pred_context = None

            comparison_imgs = [
                add_label(vcat(*context_img), "Context"),
            ]
            if rgb_pred_context is not None:
                comparison_imgs.append(add_label(vcat(*rgb_pred_context), "Context (prediction)"))
            comparison_imgs.extend([
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)")
            ])
            if rgb_pred_template is not None:
                comparison_imgs.append(add_label(vcat(*rgb_pred_template), "Target (template)"))
            if rgb_pred_img is not None:
                comparison_imgs.append(add_label(vcat(*rgb_pred_img), "Target (img)"))
            comparison = hcat(*comparison_imgs)

            self.logger.log_image(
                "inputs",
                [prep_image(add_border(hcat(comparison)))],
                step=self.global_step,
                caption=batch["scene"][:1],
            )

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, gall_template, gall_rgb = self.encoder(
                batch["context"],
                self.global_step,
                return_complete_gaussians=True
            )

        # align the target pose
        if self.test_cfg.align_pose:
            output, _, updated_params = self.test_step_align(batch, gaussians)
        elif self.test_cfg.refine_gaussians:
            output, _ = self.test_step_refine_gaussians(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output, _ = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["Rs"],
                    batch["target"]["Ts"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                    cnl_Rs=batch["target"]["cnl_Rs"],
                    cnl_Ts=batch["target"]["cnl_Ts"],
                    context_cnl_Rs=batch["context"]["cnl_Rs"],
                    context_cnl_Ts=batch["context"]["cnl_Ts"],
                )
        # output = _["output_template"]

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        path.mkdir(parents=True, exist_ok=True)
        if self.test_cfg.save_image:
            for index, color, gt_color in zip(batch["target"]["index"][0], output.color[0], batch["target"]["image"][0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")
                save_image(gt_color, path / scene / f"color/{index:0>6}_gt.png")

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0], self.encoder.cfg.input_mean, self.encoder.cfg.input_std)
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

        if self.test_cfg.save_lbs_weights:
            v_in = batch["context"]["image"].shape[1]
            j = batch["target"]["Rs"].shape[2]
            colors = np.array(sns.color_palette("tab10", n_colors=j))
            colors = torch.tensor(colors, device=gall_template.lbs_weights.device, dtype=gall_template.lbs_weights.dtype)
            lbs_weights_template = (gall_template.lbs_weights * (gall_template.opacities[..., None] > 0)).reshape(b, 1, h, w, j)
            lbs_weights_rgb = (gall_rgb.lbs_weights * (gall_rgb.opacities[..., None] > 0)).reshape(b, v_in, h, w, j)
            # lbs_weights_rgb = batch["context"]["lbs_weights"].reshape(b, v_in, h, w, j).clamp(min=1e-5).log()
            lbs_weights = torch.cat([lbs_weights_template, lbs_weights_rgb], dim=1).softmax(-1)
            lbs_weights_colors = (lbs_weights @ colors).permute(0, 1, 4, 2, 3)

            lbs_weights_colors = hcat(
                vcat(*lbs_weights_colors[0][:(v_in + 1) // 2]),
                vcat(*lbs_weights_colors[0][(v_in + 1) // 2:])
            )
            save_image(lbs_weights_colors, path / f"{scene}_lbs_weights.png")

            rgbs_pred = torch.cat([
                ((gall_template.harmonics[..., 0] * 0.28209479177387814) + 0.5).reshape(b, 1, h, w, 3),
                (((gall_rgb.harmonics[..., 0] * 0.28209479177387814) + 0.5) * gall_rgb.opacities[..., None]).reshape(b, v_in, h, w, 3),
            ], dim=1).permute(0, 1, 4, 2, 3)

            rgbs = hcat(
                vcat(*rgbs_pred[0][:(v_in + 1) // 2]),
                vcat(*rgbs_pred[0][(v_in + 1) // 2:])
            )
            save_image(rgbs, path / f"{scene}_rgbs.png")

            if self.test_cfg.align_pose:
                if not self.test_cfg.align_pose:
                    updated_params = {
                        "Rs": batch["target"]["Rs"],
                        "Ts": batch["target"]["Ts"],
                        "extrinsics": batch["target"]["extrinsics"],
                        "intrinsics": batch["target"]["intrinsics"],
                    }
                pts3d = batch["context"]["template_3d"][0].reshape(-1, 3)
                lbs_weights = batch["context"]["template_lbs_weights"][0].reshape(-1, batch["context"]["template_lbs_weights"].shape[-1])
                valid = batch["context"]["template_mask"][0].reshape(-1) > 0.001

                pts3d = pts3d[valid]
                lbs_weights = lbs_weights[valid]

                np.savez(path / f"{scene}_pose_old", Rs=batch["target"]["Rs"][0].cpu().numpy(), Ts=batch["target"]["Ts"][0].cpu().numpy())
                np.savez(path / f"{scene}_pose_new", Rs=updated_params["Rs"][0].cpu().numpy(), Ts=updated_params["Ts"][0].cpu().numpy())

                colors = []
                for i, (index, color, gt_color) in enumerate(zip(batch["target"]["index"][0], output.color[0], batch["target"]["image"][0])):
                    pts3d_warped = apply_lbs_to_means(pts3d[None], updated_params["Rs"][:, i], updated_params["Ts"][:, i],
                                                      lbs_weights[None])
                    pts3d_warped = torch.cat([pts3d_warped, torch.ones_like(pts3d_warped[..., :1])], dim=-1)
                    pts3d_cam = batch["target"]["intrinsics"][:, i] @ (updated_params["extrinsics"].inverse()[:, i, :3] @ pts3d_warped.permute(0, 2, 1))
                    pts3d_cam = pts3d_cam.permute(0, 2, 1)[0].detach().cpu().numpy()
                    # save_image(color, path / scene / f"color/{index:0>6}.png")
                    # save_image(gt_color, path / scene / f"color/{index:0>6}_gt.png")
                    color = (color.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8).copy()
                    pts_color = np.zeros_like(color)
                    for pt in pts3d_cam:
                        x, y = pt[:2] / pt[2]
                        x, y = int(x * w), int(y * h)
                        cv2.circle(pts_color, (x, y), 1, (0, 0, 255), -1)
                    colors.append((pts_color * 0.5 + color * 0.5).astype(np.uint8))
                colors = np.concatenate(colors, axis=1)
                Image.fromarray(colors).save(path / f"{scene}_pose.png")

        if self.test_cfg.save_canonical:
            center = np.array([0, -0.5, 0])

            cam_front = torch.tensor([
                [1, 0, 0, center[0]],
                [0, -1, 0, center[1]],
                [0, 0, -1, center[2] + 2],
                [0, 0, 0, 1],
            ])

            cam_back = torch.tensor([
                [-1, 0, 0, center[0]],
                [0, -1, 0, center[1]],
                [0, 0, 1, center[2] - 2],
                [0, 0, 0, 1],
            ])

            cam_right = torch.tensor([
                [0, 0, -1, center[0] + 2],
                [0, -1, 0, center[1]],
                [-1, 0, 0, center[2]],
                [0, 0, 0, 1],
            ])

            cam_left = torch.tensor([
                [0, 0, 1, center[0] - 2],
                [0, -1, 0, center[1]],
                [1, 0, 0, center[2]],
                [0, 0, 0, 1],
            ])

            extrinsics = torch.stack([cam_front, cam_back, cam_right, cam_left], 0).float().cuda()

            extrinsics = repeat(
                extrinsics,
                "v i j -> b v i j",
                b=b,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=extrinsics.shape[1],
            )
            intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 0.8
            intrinsics[..., 0, 2] = 0.5
            intrinsics[..., 1, 2] = 0.5244
            near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=extrinsics.shape[1])
            far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=extrinsics.shape[1])

            Rs = repeat(batch["target"]["Rs_tpose"][:, 0], "b j x y -> b v j x y", v=extrinsics.shape[1])
            Ts = repeat(batch["target"]["Ts_tpose"][:, 0], "b j x -> b v j x", v=extrinsics.shape[1])
            Rs[:] = 0.
            Rs[:, :, :, 0, 0] = Rs[:, :, :, 1, 1] = Rs[:, :, :, 2, 2] = 1.
            Ts[:] = 0.

            cnl_Rs = repeat(batch["target"]["cnl_Rs"][:, 0], "b j x y -> b v j x y", v=extrinsics.shape[1])
            cnl_Ts = repeat(batch["target"]["cnl_Ts"][:, 0], "b j x -> b v j x", v=extrinsics.shape[1])
            cnl_Rs[:] = 0.
            cnl_Rs[:, :, :, 0, 0] = cnl_Rs[:, :, :, 1, 1] = cnl_Rs[:, :, :, 2, 2] = 1.
            cnl_Ts[:] = 0.

            output, _ = self.decoder.forward(
                gaussians,
                extrinsics,
                intrinsics,
                Rs,
                Ts,
                near,
                far,
                (h, w),
                bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                cnl_Rs=cnl_Rs,
                cnl_Ts=cnl_Ts
            )
            canonical = hcat(
                vcat(*output.color[0][:2]),
                vcat(*output.color[0][2:])
            )
            save_image(canonical, path / f"{scene}_canonical.png")

            v_max = gaussians.idx[..., 0].max().item() + 1
            for v in range(v_max):
                gaussians_v = copy.deepcopy(gaussians)
                gaussians_v.opacities[gaussians_v.idx[..., 0] != v] = 0.
                output, _ = self.decoder.forward(
                    gaussians_v,
                    extrinsics,
                    intrinsics,
                    Rs,
                    Ts,
                    near,
                    far,
                    (h, w),
                    bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                    cnl_Rs=cnl_Rs,
                    cnl_Ts=cnl_Ts
                )
                canonical = hcat(
                    vcat(*output.color[0][:2]),
                    vcat(*output.color[0][2:])
                )
                save_image(canonical, path / f"{scene}_canonical_{v}.png")

        if self.test_cfg.save_gaussians:
            h_template, w_template = batch["context"]["template_mask"].shape[-2:]
            _, v, h, w = batch["context"]["mask"].shape
            np.savez(path / f"{scene}_gaussians_0",
                    image=batch["context"]["template_3d"][0].detach().cpu().numpy(),
                    mask=batch["context"]["template_mask"][0].detach().cpu().numpy(),
                    rgb=gall_template.harmonics[0].reshape(h_template, w_template, 3, -1)[..., 0].detach().cpu().numpy(),
                    pts3d=gall_template.means[0].reshape(h_template, w_template, 3).detach().cpu().numpy(),
                    opacties=gall_template.opacities[0].reshape(h_template, w_template).detach().cpu().numpy(),
                    )
            rgbs = gall_rgb.harmonics[0].reshape(v, h, w, 3, -1)[..., 0].detach().cpu().numpy()
            pts3d = gall_rgb.means[0].reshape(v, h, w, 3).detach().cpu().numpy()
            opacities = gall_rgb.opacities[0].reshape(v, h, w).detach().cpu().numpy()
            for vi in range(v):
                np.savez(path / f"{scene}_gaussians_{vi+1}",
                        image=context_img[vi].permute(1, 2, 0).detach().cpu().numpy(),
                        mask=batch["context"]["mask"][0, vi].detach().cpu().numpy(),
                        rgb=rgbs[vi],
                        pts3d=pts3d[vi],
                        opacities=opacities[vi],
                        )

        if self.test_cfg.pose_seq is not None:
            extrinsics, intrinsics, Rs, Ts, cnl_Rs, cnl_Ts, (h, w) = self.load_pose_seq(self.test_cfg.pose_seq)
            num_frames = extrinsics.shape[1]
            frames_per_batch = 20

            images = []
            for start_frame in range(0, num_frames, frames_per_batch):
                num_samples = min(frames_per_batch, num_frames - start_frame)
                near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_samples)
                far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_samples)
                output, _ = self.decoder.forward(
                    gaussians,
                    extrinsics[:, start_frame:start_frame+num_samples], intrinsics[:, start_frame:start_frame+num_samples],
                    Rs[:, start_frame:start_frame+num_samples], Ts[:, start_frame:start_frame+num_samples],
                    near, far,
                    (h, w), "depth",
                    bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                    cnl_Rs=cnl_Rs[:, start_frame:start_frame+num_samples],
                    cnl_Ts=cnl_Ts[:, start_frame:start_frame+num_samples],
                )
                images.extend([
                    rgb
                    for rgb in output.color[0].detach().cpu()
                ])

            video = torch.stack(images)
            video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
            value = wandb.Video(video[None], fps=10, format="mp4")

            tensor = value._prepare_video(value.data)
            clip = mpy.ImageSequenceClip(list(tensor), fps=20)
            clip.write_videofile(
                str(path / f"{scene}_novel_pose.mp4"), logger=None
            )

    def test_step_align(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        if gaussians.conf is not None:
            print(self.encoder.cfg.conf_threshold)
            gaussians.opacities *= gaussians.conf > self.encoder.cfg.conf_threshold

        b, v, _, h, w = batch["target"]["image"].shape
        _, _, j, _, _ = batch["target"]["Rs"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            if self.test_cfg.align_human_pose != "disabled":
                human_rot_delta = nn.Parameter(torch.zeros([b, 1, j, 3], requires_grad=True, device=self.device))
                human_trans_delta = nn.Parameter(torch.zeros([b, 1, j, 3], requires_grad=True, device=self.device))
                human_opt_params = []
                human_opt_params.append(
                    {
                        "params": [human_rot_delta],
                        "lr": self.test_cfg.human_rot_opt_lr,
                    }
                )
                human_opt_params.append(
                    {
                        "params": [human_trans_delta],
                        "lr": self.test_cfg.human_trans_opt_lr,
                    }
                )
                human_pose_optimizer = torch.optim.Adam(human_opt_params)

            if self.test_cfg.align_human_pose == "context":
                # optimize camera poses and human poses with context images

                new_batch = {
                    **batch,
                    "target": {
                        **batch["context"],
                        "image": inverse_normalize(batch["context"]["image"], self.encoder.cfg.input_mean,
                                                   self.encoder.cfg.input_std),
                    }
                }

                cam_rot_delta = nn.Parameter(torch.zeros([b, new_batch["target"]["image"].shape[1], 3], requires_grad=True, device=self.device))
                cam_trans_delta = nn.Parameter(torch.zeros([b, new_batch["target"]["image"].shape[1], 3], requires_grad=True, device=self.device))

                opt_params = []
                opt_params.append(
                    {
                        "params": [cam_rot_delta],
                        "lr": self.test_cfg.rot_opt_lr,
                    }
                )
                opt_params.append(
                    {
                        "params": [cam_trans_delta],
                        "lr": self.test_cfg.trans_opt_lr,
                    }
                )
                pose_optimizer = torch.optim.Adam(opt_params)

                extrinsics = new_batch["target"]["extrinsics"].clone()
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()
                    human_pose_optimizer.zero_grad()

                    Rs = (pytorch3d.transforms.so3_exp_map(human_rot_delta.reshape(-1, 3))).reshape(
                        *human_rot_delta.shape, 3) @ new_batch["target"]["Rs"]
                    Ts = human_trans_delta + new_batch["target"]["Ts"]
                    output, _ = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        new_batch["target"]["intrinsics"],
                        Rs,
                        Ts,
                        new_batch["target"]["near"],
                        new_batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        bgcolor=new_batch["bgcolor"] if "bgcolor" in new_batch else None,
                        cnl_Rs=batch["target"]["cnl_Rs"],
                        cnl_Ts=batch["target"]["cnl_Ts"],
                        context_cnl_Rs=batch["context"]["cnl_Rs"],
                        context_cnl_Ts=batch["context"]["cnl_Ts"],
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        if loss_fn.name == "lpips" or loss_fn.name == 'mse':
                            loss = loss_fn.forward(output, new_batch, gaussians, self.global_step)
                            total_loss = total_loss + loss

                    total_loss.backward()
                    human_pose_optimizer.step()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

                human_rot_delta.requires_grad = False
                human_trans_delta.requires_grad = False

                cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
                cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

                opt_params = []
                opt_params.append(
                    {
                        "params": [cam_rot_delta],
                        "lr": self.test_cfg.rot_opt_lr,
                    }
                )
                opt_params.append(
                    {
                        "params": [cam_trans_delta],
                        "lr": self.test_cfg.trans_opt_lr,
                    }
                )
                pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    if self.test_cfg.align_human_pose == "target":
                        human_pose_optimizer.zero_grad()

                    if self.test_cfg.align_human_pose != "disabled":
                        Rs = (pytorch3d.transforms.so3_exp_map(human_rot_delta.reshape(-1, 3))).reshape(*human_rot_delta.shape, 3) @ batch["target"]["Rs"]
                        Ts = human_trans_delta + batch["target"]["Ts"]
                    else:
                        Rs = batch["target"]["Rs"]
                        Ts = batch["target"]["Ts"]
                    output, _ = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        # batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        Rs,
                        Ts,
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                        cnl_Rs=batch["target"]["cnl_Rs"],
                        cnl_Ts=batch["target"]["cnl_Ts"],
                        context_cnl_Rs=batch["context"]["cnl_Rs"],
                        context_cnl_Ts=batch["context"]["cnl_Ts"],
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        if loss_fn.name == "lpips" or loss_fn.name == 'mse':
                            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                            total_loss = total_loss + loss

                    total_loss.backward()
                    if self.test_cfg.align_human_pose == "target":
                        human_pose_optimizer.step()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render Gaussians.
        if self.test_cfg.align_human_pose != "disabled":
            human_rot_delta.requires_grad = False
            human_trans_delta.requires_grad = False
            Rs = (pytorch3d.transforms.so3_exp_map(human_rot_delta.reshape(-1, 3))).reshape(*human_rot_delta.shape, 3) @ \
                 batch["target"]["Rs"]
            Ts = human_trans_delta + batch["target"]["Ts"]
        else:
            Rs = batch["target"]["Rs"]
            Ts = batch["target"]["Ts"]

        output, _ = self.decoder.forward(
            gaussians,
            extrinsics,
            # batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            Rs,
            Ts,
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
            cnl_Rs=batch["target"]["cnl_Rs"],
            cnl_Ts=batch["target"]["cnl_Ts"],
            context_cnl_Rs=batch["context"]["cnl_Rs"],
            context_cnl_Ts=batch["context"]["cnl_Ts"],
        )
        # output = _["output_template"]

        updated_params = {
            "extrinsics": extrinsics.detach(),
            "Rs": Rs.detach(),
            "Ts": Ts.detach(),
        }

        return output, _, updated_params

    def test_step_refine_gaussians(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        n = gaussians.means.shape[1]
        with torch.inference_mode(False):
            new_batch = {
                **batch,
                "target": {
                    **batch["context"],
                    "image": inverse_normalize(batch["context"]["image"], self.encoder.cfg.input_mean, self.encoder.cfg.input_std)
                },
            }

            gaussians_scale = nn.Parameter(torch.ones([b, n, 3], requires_grad=True, device=self.device) * 1e-7)
            gaussians_rot = torch.zeros([b, n, 4], device=self.device)
            gaussians_rot[..., -1] = 1.
            gaussians_rot.requires_grad = True
            gaussians_rot = nn.Parameter(gaussians_rot)

            opt_params = []
            opt_params.append(
                {
                    "params": [gaussians_scale],
                    "lr": self.test_cfg.gaussian_scale_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [gaussians_rot],
                    "lr": self.test_cfg.gaussian_rot_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            new_gaussians = Gaussians(
                gaussians.means.clone(),
                gaussians.covariances.clone(),
                gaussians.harmonics.clone(),
                gaussians.opacities.clone(),
                gaussians.lbs_weights.clone(),
                gaussians.idx.clone(),
            )

            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.refine_gaussians_steps):
                    pose_optimizer.zero_grad()

                    new_gaussians.covariances = build_covariance(gaussians_scale, gaussians_rot)
                    output, _ = self.decoder.forward(
                        new_gaussians,
                        new_batch["target"]["extrinsics"],
                        new_batch["target"]["intrinsics"],
                        new_batch["target"]["Rs"],
                        new_batch["target"]["Ts"],
                        new_batch["target"]["near"],
                        new_batch["target"]["far"],
                        (h, w),
                        bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                        cnl_Rs=batch["target"]["cnl_Rs"],
                        cnl_Ts=batch["target"]["cnl_Ts"],
                        context_cnl_Rs=batch["context"]["cnl_Rs"],
                        context_cnl_Ts=batch["context"]["cnl_Ts"],
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        if not (loss_fn.name == 'mse' or loss_fn.name == 'lpips'):
                            continue
                        loss = loss_fn.forward(output, new_batch, gaussians, self.global_step)
                        total_loss = total_loss + loss
                    # print(total_loss)

                    total_loss.backward()
                    pose_optimizer.step()
        torch.inference_mode()

        # Render Gaussians.
        new_gaussians.covariances = build_covariance(gaussians_scale, gaussians_rot).clone()
        new_gaussians.covariances.requires_grad = False
        output, _ = self.decoder.forward(
            new_gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["Rs"],
            batch["target"]["Ts"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
            cnl_Rs=batch["target"]["cnl_Rs"],
            cnl_Ts=batch["target"]["cnl_Ts"],
            context_cnl_Rs=batch["context"]["cnl_Rs"],
            context_cnl_Ts=batch["context"]["cnl_Ts"],
        )

        return output, _

    def load_pose_seq(self, pose_seq_path):
        device = "cuda"

        intrinsics_seq = []
        extrinsics_seq = []
        Rs_seq = []
        Ts_seq = []
        cnl_Rs_seq = []
        cnl_Ts_seq = []

        betas = None
        for filename in sorted(os.listdir(pose_seq_path)):
            meta = json.load(open(os.path.join(pose_seq_path, filename)))
            if betas is None:
                betas = np.array(meta["betas"])

            trans = np.array(meta["trans"])
            global_orient = np.array(meta["root_pose"])
            body_pose = np.array(meta["body_pose"])
            left_hand_pose = np.array(meta["lhand_pose"])
            right_hand_pose = np.array(meta["rhand_pose"])
            jaw_pose = np.array(meta["jaw_pose"])
            leye_pose = np.array(meta["leye_pose"])
            reye_pose = np.array(meta["reye_pose"])

            w, h = meta["img_size_wh"]
            intrinsics = np.array([
                [meta["focal"][0], 0, meta["princpt"][0]],
                [0, meta["focal"][1], meta["princpt"][1]],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics[0] /= w
            intrinsics[1] /= h

            with torch.no_grad():
                output = self.smplx_model(
                    betas=torch.tensor(betas)[None].float().to(device),
                    return_full_pose=True,
                )
            tpose_joints = output.joints.detach().cpu().numpy()[0, :55]
            tpose_joints = self.template_tpose_joints.detach().cpu().numpy()

            Rh = global_orient
            Th = trans
            Th = Th + tpose_joints[0] - _rvec_to_rmtx(Rh) @ tpose_joints[0]

            extrinsics = apply_global_tfm_to_camera(
                E=np.eye(4, dtype=np.float32),
                Rh=Rh,
                Th=Th)

            with torch.no_grad():
                output = self.smplx_model(
                    body_pose=torch.tensor(body_pose)[None].float().to(device),
                    betas=torch.tensor(betas)[None].float().to(device),
                    left_hand_pose=torch.tensor(left_hand_pose)[None].float().to(device),
                    right_hand_pose=torch.tensor(right_hand_pose)[None].float().to(device),
                    jaw_pose=torch.tensor(jaw_pose)[None].float().to(device),
                    leye_pose=torch.tensor(leye_pose)[None].float().to(device),
                    reye_pose=torch.tensor(reye_pose)[None].float().to(device),
                    return_full_pose=True,
                )
            full_pose = (output.full_pose - 0 * self.pose_mean).detach().cpu().numpy().reshape(55, 3)
            full_pose[0] = 0.

            cnl_gtfms = get_canonical_global_tfms(tpose_joints, use_smplx=True)
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                full_pose, tpose_joints, use_smplx=True
            )
            global_Rs, global_Ts = get_global_RTs(
                cnl_gtfms, dst_Rs, dst_Ts,
                use_smplx=True)

            cnl_Rs, cnl_Ts = get_canonical_tfms(self.template_tpose_joints, torch.tensor(tpose_joints), use_smplx=True)

            intrinsics_seq.append(intrinsics)
            extrinsics_seq.append(extrinsics)
            Rs_seq.append(global_Rs)
            Ts_seq.append(global_Ts)
            cnl_Rs_seq.append(cnl_Rs)
            cnl_Ts_seq.append(cnl_Ts)

        extrinsics_seq = torch.tensor(np.stack(extrinsics_seq)).to(device)[None].float().inverse()
        intrinsics_seq = torch.tensor(np.stack(intrinsics_seq)).to(device)[None].float()
        Rs_seq = torch.tensor(np.stack(Rs_seq)).to(device)[None].float()
        Ts_seq = torch.tensor(np.stack(Ts_seq)).to(device)[None].float()
        cnl_Rs_seq = torch.stack(cnl_Rs_seq).to(device)[None].float()
        cnl_Ts_seq = torch.stack(cnl_Ts_seq).to(device)[None].float()
        return extrinsics_seq, intrinsics_seq, Rs_seq, Ts_seq, cnl_Rs_seq, cnl_Ts_seq, (h, w)


    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        output, output_aux = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["Rs"],
            batch["target"]["Ts"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
            bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
            context_extrinsics=batch["context"]["extrinsics"],
            context_intrinsics=batch["context"]["intrinsics"],
            context_Rs=batch["context"]["Rs"],
            context_Ts=batch["context"]["Ts"],
            context_near=batch["context"]["near"],
            context_far=batch["context"]["far"],
            cnl_Rs=batch["target"]["cnl_Rs"],
            cnl_Ts=batch["target"]["cnl_Ts"],
            context_cnl_Rs=batch["context"]["cnl_Rs"],
            context_cnl_Ts=batch["context"]["cnl_Ts"],
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        if "output_img" in output_aux:
            rgb_pred_img = output_aux["output_img"].color[0]
        else:
            rgb_pred_img = None

        if "output_template" in output_aux:
            rgb_pred_template = output_aux["output_template"].color[0]
        else:
            rgb_pred_template = None

        if "output_context" in output_aux:
            rgb_pred_context = output_aux["output_context"].color[0]
        else:
            rgb_pred_context = None

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # direct depth from gaussian means (used for visualization only)
        if "depth" in visualization_dump:
            gaussian_means = visualization_dump["depth"][0].squeeze(-1).squeeze(-1)
            if gaussian_means.shape[-1] == 3:
                gaussian_means = gaussian_means.mean(dim=-1)

            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0], self.encoder.cfg.input_mean, self.encoder.cfg.input_std)
            context_img_depth = vis_depth_map(gaussian_means)
            context = []
            for i in range(context_img.shape[0]):
                context.append(context_img[i])
                context.append(context_img_depth[i])
            comparison_imgs = [
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*depth_pred), "Depth (Prediction)"),
            ]
            if rgb_pred_template is not None:
                comparison_imgs.append(add_label(vcat(*rgb_pred_template), "Target (template)"))
            if rgb_pred_img is not None:
                comparison_imgs.append(add_label(vcat(*rgb_pred_img), "Target (img)"))
            if rgb_pred_context is not None:
                comparison_imgs.append(add_label(vcat(*rgb_pred_context), "Context (prediction)"))
            comparison = hcat(*comparison_imgs)

            if self.distiller is not None:
                with torch.no_grad():
                    pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
                depth1, depth2 = pseudo_gt1['pts3d'][..., -1], pseudo_gt2['pts3d'][..., -1]
                conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
                depth_dust = torch.cat([depth1, depth2], dim=0)
                depth_dust = vis_depth_map(depth_dust)
                conf_dust = torch.cat([conf1, conf2], dim=0)
                conf_dust = confidence_map(conf_dust)
                dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
                comparison = hcat(add_label(vcat(*dust_vis), "Context"), comparison)

            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
        else:
            context_img = inverse_normalize(batch["context"]["image"][0], self.encoder.cfg.input_mean, self.encoder.cfg.input_std)
            context = []
            for i in range(context_img.shape[0]):
                context.append(context_img[i])
            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )


        # # Render projections and construct projection image.
        # # These are disabled for now, since RE10k scenes are effectively unbounded.
        # projections = hcat(
        #         *render_projections(
        #             gaussians,
        #             256,
        #             extra_label="",
        #         )[0]
        #     )
        # self.logger.log_image(
        #     "projection",
        #     [prep_image(add_border(projections))],
        #     step=self.global_step,
        # )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #     self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_canonical(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "canonical", num_frames=60)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        # if v != 2:
        #     return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=30)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics_all = []
            intrinsics_all = []
            for vi in range(v):
                extrinsics = interpolate_extrinsics(
                    batch["context"]["extrinsics"][0, vi],
                    (
                        batch["context"]["extrinsics"][0, (vi + 1) % v]
                    ),
                    t,
                )
                intrinsics = interpolate_intrinsics(
                    batch["context"]["intrinsics"][0, vi],
                    (
                        batch["context"]["intrinsics"][0, (vi + 1) % v]
                    ),
                    t,
                )
                extrinsics_all.append(extrinsics)
                intrinsics_all.append(intrinsics)
            return torch.cat(extrinsics_all, dim=0)[None], torch.cat(intrinsics_all, dim=0)[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb", num_frames=30, loop_reverse=False)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_freeview(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            ROT_CAM_PARAMS = {'rotate_axis': 'y', 'inv_angle': False}
            extrinsics = []
            for ti, _ in enumerate(t):
                E = rotate_camera_by_frame_idx(
                    extrinsics=batch["context"]["extrinsics"][0, 0].inverse().detach().cpu().numpy(),
                    frame_idx=ti,
                    period=t.shape[0],
                    **ROT_CAM_PARAMS)
                extrinsics.append(E)
            extrinsics = torch.tensor(np.stack(extrinsics), dtype=torch.float32, device=batch["context"]["extrinsics"].device)[None]
            intrinsics = repeat(batch["context"]["intrinsics"][:, 0], "b i j -> b v i j", v=t.shape[0])
            return extrinsics.inverse(), intrinsics

        scene = batch["scene"][0]
        return self.render_video_generic(
            batch,
            trajectory_fn,
            f"freeview_{scene}",
            num_frames=100,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=extrinsics.shape[1])
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=extrinsics.shape[1])
        Rs = repeat(batch["context"]["Rs"][:, 0], "b j x y -> b v j x y", v=extrinsics.shape[1])
        Ts = repeat(batch["context"]["Ts"][:, 0], "b j x -> b v j x", v=extrinsics.shape[1])
        cnl_Rs = repeat(batch["context"]["cnl_Rs"][:, 0], "b j x y -> b v j x y", v=extrinsics.shape[1])
        cnl_Ts = repeat(batch["context"]["cnl_Ts"][:, 0], "b j x -> b v j x", v=extrinsics.shape[1])
        images_all = []
        for start_idx in range(0, extrinsics.shape[1], 20):
            end_idx = min(start_idx + 20, extrinsics.shape[1])
            output, _ = self.decoder.forward(
                gaussians,
                extrinsics[:, start_idx:end_idx], intrinsics[:, start_idx:end_idx],
                Rs[:, start_idx:end_idx], Ts[:, start_idx:end_idx],
                near[:, start_idx:end_idx], far[:, start_idx:end_idx],
                (h, w), "depth",
                bgcolor=batch["bgcolor"] if "bgcolor" in batch else None,
                cnl_Rs=cnl_Rs[:, start_idx:end_idx], cnl_Ts=cnl_Ts[:, start_idx:end_idx],
                context_cnl_Rs=batch["context"]["cnl_Rs"],
                context_cnl_Ts=batch["context"]["cnl_Ts"],
            )
            images = [
                rgb.detach()
                for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
            ]
            images_all.extend(images)

        video = torch.stack(images_all)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=10, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        # try:
        #     wandb.log(visualizations)
        # except Exception:
        #     assert isinstance(self.logger, LocalLogger)
        for key, value in visualizations.items():
            tensor = value._prepare_video(value.data)
            clip = mpy.ImageSequenceClip(list(tensor), fps=20)
            dir = LOG_PATH / key
            dir.mkdir(exist_ok=True, parents=True)
            clip.write_videofile(
                str(dir / f"{name}.mp4"), logger=None
            )
            print(str(dir / f"{name}.mp4"))

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]