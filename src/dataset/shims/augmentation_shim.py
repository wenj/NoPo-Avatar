import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews
from ...misc.body_utils import SMPLX_JOINT_FLIP, SMPL_JOINT_FLIP, SMPLX_BONE_FLIP, SMPL_BONE_FLIP


def reflect_cameras(
    extrinsics: Float[Tensor, "*batch 4 4"],
    intrinsics: Float[Tensor, "*batch 3 3"]
) -> tuple[Float[Tensor, "*batch 4 4"], Float[Tensor, "*batch 4 4"]]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    new_extrinsics = reflect @ extrinsics @ reflect
    new_intrinsics = intrinsics.clone()
    new_intrinsics[..., 0, 2] = 1 - new_intrinsics[..., 0, 2]
    return new_extrinsics, new_intrinsics


def reflect_poses(
    Rs: Float[Tensor, "*batch joint 3 3"],
    Ts: Float[Tensor, "*batch joint 3"],
    flip_joint: bool = True,
    use_smplx: bool = True
) -> tuple[Float[Tensor, "*batch joint 3 3"], Float[Tensor, "*batch joint 3"]]:
    if flip_joint:
        if use_smplx:
            flip = SMPLX_JOINT_FLIP
        else:
            flip = SMPL_JOINT_FLIP
    else:
        if use_smplx:
            flip = SMPLX_BONE_FLIP
        else:
            flip = SMPL_BONE_FLIP
    new_Rs = Rs[..., flip, :, :]
    new_Ts = Ts[..., flip, :]

    reflect = torch.eye(3, dtype=torch.float32, device=Rs.device)
    reflect[0, 0] = -1
    new_Rs = reflect @ new_Rs @ reflect
    new_Ts = (reflect @ new_Ts.unsqueeze(-1)).squeeze(-1)
    return new_Rs, new_Ts


def reflect_views(views: AnyViews) -> AnyViews:
    Rs, Ts = reflect_poses(views["Rs"], views["Ts"], use_smplx=views["use_smplx"])
    Rs_tpose, Ts_tpose = reflect_poses(views["Rs_tpose"], views["Ts_tpose"], use_smplx=views["use_smplx"])
    cnl_Rs, cnl_Ts = reflect_poses(views["cnl_Rs"], views["cnl_Ts"], flip_joint=False, use_smplx=views["use_smplx"])
    extrinsics, intrinsics = reflect_cameras(views["extrinsics"], views["intrinsics"])
    new_views = {
        **views,
        "image": views["image"].flip(-1),
        "mask": views["mask"].flip(-1),
        # "lbs_weights": views["lbs_weights"].flip(-2),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "Rs": Rs,
        "Ts": Ts,
        "Rs_tpose": Rs_tpose,
        "Ts_tpose": Ts_tpose,
        "cnl_Rs": cnl_Rs,
        "cnl_Ts": cnl_Ts,
    }
    if "lbs_weights" in views:
        lbs_weights = views["lbs_weights"].flip(-2)
        joint_flip = SMPLX_JOINT_FLIP if views["use_smplx"] else SMPL_JOINT_FLIP
        lbs_weights = lbs_weights[..., joint_flip]
        new_views["lbs_weights"] = lbs_weights
    return new_views


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }