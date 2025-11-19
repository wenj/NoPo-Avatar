import json
import subprocess
import sys
import os
import pickle
from pathlib import Path
from typing import Literal, TypedDict, Optional

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import seaborn as sns
import trimesh

import cv2
from PIL import Image
from io import BytesIO

import nvdiffrast
import nvdiffrast.torch

from ..misc.body_utils import get_canonical_global_tfms, get_global_RTs, body_pose_to_body_RTs, apply_global_tfm_to_camera, apply_lbs_to_means

DEBUG = False
THuman21 = False
RASTERIZE_LBS_WEIGHTS = True

INPUT_DIR = Path("datasets/thuman")
if THuman21:
    OUTPUT_DIR = Path("datasets/thuman2.1")
else:
    OUTPUT_DIR = Path("datasets/thuman2.0")

TRAIN_FRAME_ORDERS = []
for i in range(16):
    TRAIN_FRAME_ORDERS.append(i)
    TRAIN_FRAME_ORDERS.append(16 + i * 3)
    TRAIN_FRAME_ORDERS.append(16 + i * 3 + 1)
    TRAIN_FRAME_ORDERS.append(16 + i * 3 + 2)
TEST_FRAME_ORDERS = [0, 1, 2, 3, 4, 5]


# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(5e7)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    if THuman21 and stage == "train":
        with open(f'datasets/thuman/thuman2.1_train.json') as f:
            keys = json.load(f)
        return keys
    with open(f'datasets/thuman/thuman2.0_{stage}.json') as f:
        keys = json.load(f)
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[str, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {path.stem: load_raw(path) for path in example_path.iterdir() if path.name.endswith('.png')}


def clip_T_world(xyzs_world, K, E, H, W):
    xyzs = torch.cat([xyzs_world, torch.ones_like(xyzs_world[..., 0:1, :])], dim=-2)
    K_expand = torch.zeros_like(E)
    znear, zfar = 1e-3, 1e3
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    K_expand[:, 0, 0] = 2.0 * fx / W
    K_expand[:, 1, 1] = 2.0 * fy / H
    K_expand[:, 0, 2] = 2.0 * cx / W - 1.0
    K_expand[:, 1, 2] = 2.0 * cy / H - 1.0
    K_expand[:, 2, 2] = (zfar + znear) / (zfar - znear)
    K_expand[:, 3, 2] = 1.
    K_expand[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)
    return (K_expand @ E @ xyzs).permute(0, 2, 1)


def rasterize_lbs_weights(rasterize_context, xyz, lbs_weights, K, E, faces, resolution):
    # now assume resolution is [H, W]
    xyz = torch.tensor(xyz).cuda()[None]
    lbs_weights = torch.tensor(lbs_weights).cuda()[None]
    K = torch.tensor(K).cuda()[None]
    E = torch.tensor(np.concatenate([E.reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)).cuda()[None]
    faces = torch.tensor(faces.astype(int)).cuda().type(torch.int32)
    NP = xyz.shape[1]

    resolution_new_0 = (resolution[0] // 8 + ((resolution[0] % 8) > 0)) * 8
    resolution_new_1 = (resolution[1] // 8 + ((resolution[1] % 8) > 0)) * 8

    xyzs_clip = clip_T_world(xyz.permute(0, 2, 1).float(), K.float(), E.float(), resolution_new_0,
                             resolution_new_1).contiguous()

    # the resolution for nvdiffrast is [H, W]
    outputs, _ = nvdiffrast.torch.rasterize(rasterize_context, xyzs_clip, faces,
                                            [resolution_new_0, resolution_new_1])

    lbs_weights, _ = nvdiffrast.torch.interpolate(lbs_weights, outputs, faces, [resolution_new_0, resolution_new_1])
    return lbs_weights[0].contiguous().detach().cpu()


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, "camera"]
    cameras: Float[Tensor, "camera entry"]
    poses: Float[Tensor, "pose entry"]
    poses_tpose: Float[Tensor, "pose entry"]
    supervisions: Optional[Float[Tensor, "pose h w entry"]]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    w2c_opencv = np.linalg.inv(c2w)
    return w2c_opencv


def load_metadata(camera_path: Path, canonical_path: Path, pose_path: Path, scene_name, split) -> Metadata:
    url = ""

    # FIXME: igore k1, k2, p1, p2, is this proper?
    w = 1024
    h = 1024

    with open(camera_path, 'rb') as f:
        camera_infos = pickle.load(f)
    with open(canonical_path, 'rb') as f:
        canonical_infos = pickle.load(f)
    with open(pose_path, 'rb') as f:
        pose_infos = pickle.load(f)

    vertex = canonical_infos['vertex']
    lbs_weights = canonical_infos['weights']
    faces = canonical_infos['faces']

    timestamps = []
    cameras = []
    poses = []
    poses_tpose = []
    tposes_joints = []
    poses_angles_all = []
    lbs_weights_imgs = []

    if split == 'train':
        frame_orders = TRAIN_FRAME_ORDERS
    else:
        frame_orders = TEST_FRAME_ORDERS

    for i, frame in enumerate(frame_orders):
        key = f'frame_{frame:06d}'
        intrinsic = camera_infos[key]['intrinsics'].astype(np.float32)
        intrinsic = [intrinsic[0, 0] / w, intrinsic[1, 1] / h, intrinsic[0, 2] / w, intrinsic[1, 2] / h, 0.0, 0.0]
        w2c = camera_infos[key]['extrinsics'].astype(np.float32)
        Rh, Th = pose_infos[key]['Rh'], pose_infos[key]['Th']
        w2c = apply_global_tfm_to_camera(
            E=w2c,
            Rh=Rh,
            Th=Th)
        w2c = w2c[:3, :]
        w2c = w2c.flatten()

        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)

        # extract number from string like "images/frame_00002.png"
        timestamps.append(frame)

        cnl_gtfms = get_canonical_global_tfms(pose_infos[key]['tpose_joints'], use_smplx=True)

        tpose_joints = pose_infos[key]['tpose_joints']
        poses_angles = pose_infos[key]['poses']

        dst_Rs, dst_Ts = body_pose_to_body_RTs(
            poses_angles, tpose_joints, use_smplx=True
        )
        global_Rs, global_Ts = get_global_RTs(
			cnl_gtfms, dst_Rs, dst_Ts,
			use_smplx=True)
        pose = np.concatenate([
            global_Rs.reshape(-1),
            global_Ts.reshape(-1)
        ])

        dst_Rs_Tpose, dst_Ts_Tpose = body_pose_to_body_RTs(
            np.zeros_like(poses_angles), tpose_joints, use_smplx=True
        )
        global_Rs_Tpose, global_Ts_Tpose = get_global_RTs(
            cnl_gtfms, dst_Rs_Tpose, dst_Ts_Tpose,
            use_smplx=True)
        pose_tpose = np.concatenate([
            global_Rs_Tpose.reshape(-1),
            global_Ts_Tpose.reshape(-1)
        ])

        poses.append(pose)
        poses_tpose.append(pose_tpose)
        tposes_joints.append(tpose_joints)
        poses_angles_all.append(poses_angles)

        if RASTERIZE_LBS_WEIGHTS:
            resolution = 1024
            rasterize_context = nvdiffrast.torch.RasterizeCudaContext(device='cuda')
            vertex_obs = apply_lbs_to_means(torch.tensor(vertex)[None], torch.tensor(global_Rs)[None],
                                   torch.tensor(global_Ts)[None], torch.tensor(lbs_weights)[None])
            vertex_obs = vertex_obs.detach().numpy()[0]
            intrinsics = camera_infos[key]['intrinsics'].astype(np.float32)
            intrinsics[:2] *= resolution / 1024
            lbs_weights_img = rasterize_lbs_weights(
                rasterize_context,
                vertex_obs,
                lbs_weights,
                intrinsics, w2c,
                faces,
                [resolution, resolution])
            os.makedirs(os.path.join(OUTPUT_DIR, f"lbs_weights_supervisions", scene_name), exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"lbs_weights_supervisions", scene_name, key)
            np.savez_compressed(path, lbs_weights=lbs_weights_img.numpy())
            lbs_weights_imgs.append(lbs_weights_img)

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    poses = torch.tensor(np.stack(poses), dtype=torch.float32)
    poses_tpose = torch.tensor(np.stack(poses_tpose), dtype=torch.float32)
    tposes_joints = torch.tensor(np.stack(tposes_joints), dtype=torch.float32)
    poses_angles_all = torch.tensor(np.stack(poses_angles_all), dtype=torch.float32)
    outputs = {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
        "poses": poses,
        "poses_tpose": poses_tpose,
        "tposes_joints": tposes_joints,
        "poses_angles_all": poses_angles_all,
    }
    return outputs


def vis_example(example, canonical_path):
    w, h = 1024, 1024
    with open(canonical_path, 'rb') as f:
        canonical_infos = pickle.load(f)
    vertex = canonical_infos['vertex']
    lbs_weights = canonical_infos['weights']
    faces = canonical_infos['faces']

    scene_name = example["key"]

    N = example["timestamps"].shape[0]
    for n in range(N):
        image = np.array(Image.open(BytesIO(example["images"][n].numpy().tobytes())))
        # image = np.zeros_like(image)

        # image_ds = np.array(Image.open(BytesIO(example["images"][n].numpy().tobytes())).resize((256, 256)))
        # mask = supervision_raw.sum(-1) != 0
        # supervision = supervision_raw.reshape(-1, 58)[mask.reshape(-1)]
        # vertex = supervision[:, :3]
        # lbs_weights = supervision[:, 3:]
        # color = image_ds.reshape(-1, 3)[mask.reshape(-1)]

        global_Rs = example["poses"][n][:55 * 3 * 3].reshape(55, 3, 3)
        global_Ts = example["poses"][n][55 * 3 * 3:].reshape(55, 3)

        vertex_obs = apply_lbs_to_means(torch.tensor(vertex)[None], global_Rs[None], global_Ts[None], torch.tensor(lbs_weights)[None])
        vertex_obs = vertex_obs.detach().numpy()[0]
        # mesh = trimesh.Trimesh(vertices=vertex_obs, faces=faces)
        vertex_obs = np.concatenate([vertex_obs, np.ones((vertex_obs.shape[0], 1))], axis=1)

        intrinsics = torch.tensor([
            [example["cameras"][n][0].item() * w, 0, example["cameras"][n][2].item() * w],
            [0, example["cameras"][n][1].item() * h, example["cameras"][n][3].item() * h],
            [0, 0, 1]
        ]).numpy()
        w2c = example["cameras"][n][6:].reshape(3, 4).numpy()

        vertex_cam = w2c.reshape(3, 4) @ vertex_obs.T
        vertex_cam = vertex_cam[:3, :].T
        vertex_im = intrinsics @ vertex_cam.T
        vertex_im = (vertex_im[:2] / vertex_im[2:]).T
        for i, (x, y) in enumerate(vertex_im.astype(int)):
            cv2.circle(image, (x, y), 2, [255, 0, 0], -1)
        print('temp/{}_{:04d}.png'.format(scene_name, n), flush=True)
        cv2.imwrite('temp/{}_{:04d}.png'.format(scene_name, n), image)


if __name__ == "__main__":
    for stage in ("train", "val", "test"):
        if stage == "train":
            path = INPUT_DIR / "train"
        elif stage == "val":
            path = INPUT_DIR / "val"
        elif stage == "test":
            path = INPUT_DIR / "val"
        keys = get_example_keys("train" if stage == "train" else "val")

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB).", flush=True
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = path / key / "images"
            mask_dir = path / key / "masks"
            canonical_metafile = path / key / "canonical_joints.pkl"
            camera_metafile = path / key / "cameras.pkl"
            pose_metafile = path / key / "mesh_infos.pkl"

            # Read images and metadata.
            images = load_images(image_dir)
            masks = load_images(mask_dir)
            example = load_metadata(camera_metafile, canonical_metafile, pose_metafile, key, stage)

            num_bytes = get_size(path / key)
            # Merge the images into the example.
            # from int to "frame_00001" format
            image_names = [f"frame_{timestamp.item():0>6}" for timestamp in example["timestamps"]]
            try:
                example["images"] = [
                    images[image_name] for image_name in image_names
                ]
                example["masks"] = [
                    masks[image_name] for image_name in image_names
                ]
            except KeyError:
                print(f"Skipping {key} because of missing images.")
                continue
            assert len(example["images"]) == len(example["timestamps"]), f"len(example['images'])={len(example['images'])}, len(example['timestamps'])={len(example['timestamps'])}"

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).", flush=True)
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

            if DEBUG:
                vis_example(example, canonical_metafile)

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)
