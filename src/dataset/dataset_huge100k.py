import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import numpy as np
import os
from smplx import SMPLX

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .shims.color_jitter_shim import apply_color_jitter_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
from ..misc.body_utils import get_canonical_tfms


@dataclass
class DatasetHuge100KCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    augment_color_jitter: bool
    relative_pose: bool
    skip_bad_shape: bool
    template_image_shape: list[int]
    load_template_uv: bool = False
    load_supervision: bool = False
    load_lbs_weights: bool = False
    sample_rate: float = 1.0


@dataclass
class DatasetHuge100KCfgWrapper:
    huge100k: DatasetHuge100KCfg


class DatasetHuge100K(IterableDataset):
    cfg: DatasetHuge100KCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetHuge100KCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        template_shape = cfg.template_image_shape[0]
        if self.cfg.load_template_uv:
            self.template_mask = torch.tensor(np.load(f'assets/templates/mask_res{template_shape}.npy')).float()
            self.template_3d = torch.tensor(np.load(f'assets/templates/xyz_res{template_shape}.npy')).float()
            self.template_lbs_weights = torch.tensor(
                np.load(f'assets/templates/lbs_weights_res{template_shape}.npy')).float()

            self.template_stds = []
            self.template_means = []

            D = self.template_lbs_weights.shape[-1]
            for d in range(D):
                valid = self.template_lbs_weights[..., d].reshape(-1) > 0
                pts = self.template_3d.reshape(-1, 3)[valid]
                weights = self.template_lbs_weights[..., d].reshape(-1)[valid]
                cov = torch.cov(pts.T, aweights=weights)
                L, V = torch.linalg.eig(cov)
                L = torch.sqrt(torch.real(L))
                V = torch.real(V)

                self.template_stds.append(V @ torch.diag(L))
                self.template_means.append(torch.sum(pts * weights[:, None], dim=0) / torch.sum(weights))

            self.template_stds = torch.stack(self.template_stds)
            self.template_means = torch.stack(self.template_means)

            MODEL_DIR = "datasets/smplx"
            smplx_model = SMPLX(model_path=os.path.join(MODEL_DIR, 'SMPLX_MALE.npz'))
            self.template_tpose_joints = smplx_model().joints.detach().cpu()[0, :55]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path, weights_only=False)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                Rs, Ts = self.convert_human_poses(example["poses"])
                Rs_tpose, Ts_tpose = self.convert_human_poses(example["poses_tpose"])

                tpose_joints = example["tposes_joints"].reshape(-1, 55, 3)
                cnl_Rs, cnl_Ts = get_canonical_tfms(self.template_tpose_joints, tpose_joints[0], use_smplx=True)
                cnl_Rs = cnl_Rs[None].repeat(tpose_joints.shape[0], 1, 1, 1)
                cnl_Ts = cnl_Ts[None].repeat(tpose_joints.shape[0], 1, 1)

                scene = example["key"]
                # if self.cfg.load_lbs_weights and self.stage == "train":
                #     lbs_weights = example["lbs_weights"]

                try:
                    context_indices, target_indices, overlap = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                context_masks = [
                    example["masks"][index.item()] for index in context_indices
                ]
                context_masks = self.convert_masks(context_masks)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
                target_masks = [
                    example["masks"][index.item()] for index in target_indices
                ]
                target_masks = self.convert_masks(target_masks)

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
                target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)

                context_images, target_images, bgcolor = self.set_bgcolor(
                    context_images, context_masks,
                    target_images, target_masks,
                    self.cfg.background_color)
                # bgcolor = torch.tensor([1.0, 1.0, 1.0], dtype=target_images.dtype)

                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if self.cfg.make_baseline_1:
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                        print(
                            f"Skipped {scene} because of baseline out of range: "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                if self.cfg.relative_pose:
                    extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "Rs": Rs[context_indices],
                        "Ts": Ts[context_indices],
                        "Rs_tpose": Rs_tpose[context_indices],
                        "Ts_tpose": Ts_tpose[context_indices],
                        "cnl_Rs": cnl_Rs[context_indices],
                        "cnl_Ts": cnl_Ts[context_indices],
                        # "lbs_weights": lbs_weights[context_indices],
                        "image": context_images,
                        "mask": context_masks,
                        "near": self.get_bound("near", len(context_indices)) / scale,
                        "far": self.get_bound("far", len(context_indices)) / scale,
                        "index": context_indices,
                        "overlap": overlap,
                        "use_smplx": True
                        # "canonical_vertices": example["canonical_vertex"],
                        # "canonical_lbs_weights": example["canonical_lbs_weights"],
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "Rs": Rs[target_indices],
                        "Ts": Ts[target_indices],
                        "Rs_tpose": Rs_tpose[target_indices],
                        "Ts_tpose": Ts_tpose[target_indices],
                        "cnl_Rs": cnl_Rs[target_indices],
                        "cnl_Ts": cnl_Ts[target_indices],
                        # "lbs_weights": lbs_weights[target_indices],
                        "image": target_images,
                        "mask": target_masks,
                        "near": self.get_bound("near", len(target_indices)) / scale,
                        "far": self.get_bound("far", len(target_indices)) / scale,
                        "index": target_indices,
                        "use_smplx": True
                    },
                    "scene": scene,
                    "bgcolor": bgcolor,
                }

                if self.cfg.load_template_uv:
                    example["context"].update({
                        "template_mask": self.template_mask,
                        "template_3d": self.template_3d,
                        "template_lbs_weights": self.template_lbs_weights,
                        "template_stds": self.template_stds,
                        "template_means": self.template_means,
                    })

                if self.cfg.load_supervision:
                    # test only; not ready
                    supervisions = []
                    for idx in context_indices:
                        supervision = np.load(f'/home/jw116/codes/generalizable_point-based-human/log/ghg_view3_subdivide_deepf_pointtransformer_iter3_fb_tf_lpips0.5_adam_lr1e-4_5e-5_nolpips_ssim_coeff1.0_lap100.0/supervision/view/scene_{scene}_frame_{idx:06d}.npy')
                        supervisions.append(supervision)
                    supervisions = np.stack([supervisions[0][0]] + [supervision[1] for supervision in supervisions])
                    supervisions = torch.from_numpy(supervisions).float()
                    example["context"]["supervisions"] = supervisions

                if self.cfg.load_lbs_weights and self.stage == "train":
                    lbs_weights_root = self.cfg.roots[0].parent / "lbs_weights_supervisions"
                    subset = scene[:11]
                    if scene[:11] in ["flux_batch1", "flux_batch2", "flux_batch3", "flux_batch4", "flux_batch5", "flux_batch6", "flux_batch8", "flux_batch9", "deepfashion"]:
                        subdir = scene[12:19]
                        subject = scene[20:]
                    else:
                        subdir = scene[12:22]
                        subject = scene[23:]
                    lbs_weights_dir = os.path.join(lbs_weights_root, subset, subdir, subject)
                    if not os.path.exists(lbs_weights_dir):
                        continue
                    lbs_weights = []
                    for idx in context_indices:
                        lbs_weights_single = np.load(
                            f'{lbs_weights_dir}/{idx:06d}.npz')['lbs_weights']
                        lbs_weights.append(lbs_weights_single)
                    example["context"].update({
                        "lbs_weights": torch.from_numpy(np.stack(lbs_weights)).float()
                    })
                    # example["context"].update({
                    #     "lbs_weights": lbs_weights[context_indices]
                    # })
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                    if self.cfg.augment_color_jitter:
                        example = apply_color_jitter_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.input_image_shape), pad=True)

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_human_poses(
        self,
        poses: Float[Tensor, "batch 660"],
    )-> tuple[
        Float[Tensor, "batch joints 3 3"],  # Rs
        Float[Tensor, "batch joints 3"],  # Ts
    ]:
        Rs = poses[:, :55 * 3 * 3]
        Ts = poses[:, 55 * 3 * 3:]
        Rs = rearrange(Rs, "b (j x y) -> b j x y", j=55, x=3, y=3)
        Ts = rearrange(Ts, "b (j x) -> b j x", j=55, x=3)
        return Rs, Ts

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_masks(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image)[0])
        return torch.stack(torch_images)

    def set_bgcolor(
        self,
        context_images: Float[Tensor, "batch 3 height width"],
        context_masks: Float[Tensor, "batch height width"],
        target_images: Float[Tensor, "batch 3 height width"],
        target_masks: Float[Tensor, "batch height width"],
        bgcolor: list[float],
    ):
        if bgcolor == [-1, -1, -1]:
            bgcolor = torch.randint(0, 255, [3], device=context_images.device) / 255.
            bgcolor = bgcolor.float()
        else:
            bgcolor = torch.tensor(bgcolor).type(context_images.dtype)
        context_images = context_images * context_masks.unsqueeze(1) + bgcolor[None, :, None, None] * (1 - context_masks.unsqueeze(1))
        target_images = target_images * target_masks.unsqueeze(1) + bgcolor[None, :, None, None] * (1 - target_masks.unsqueeze(1))
        return context_images, target_images, bgcolor

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())