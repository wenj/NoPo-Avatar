import json
import subprocess
import sys
import os
import argparse
import pickle
import imageio
from pathlib import Path
from typing import Literal, TypedDict, Optional

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import seaborn as sns
from smplx import SMPLX

import cv2
from PIL import Image
from io import BytesIO

import nvdiffrast
import nvdiffrast.torch

from ..misc.body_utils import get_canonical_global_tfms, get_global_RTs, body_pose_to_body_RTs, apply_global_tfm_to_camera, apply_lbs_to_means, _rvec_to_rmtx

DEBUG = False
RASTERIZE_LBS_WEIGHTS = True

MODEL_DIR = "datasets/smplx"

INPUT_DIR = Path("datasets/HuGe100K/all")
OUTPUT_DIR = Path("datasets/huge100k")
LBS_WEIGHTS_SAVE_DIR = Path(OUTPUT_DIR / "lbs_weights_supervisions")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(5e7)


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

    # gl_transform = torch.tensor([[1., 0, 0, 0],
    #                              [0, -1., 0, 0],
    #                              [0, 0, -1., 0],
    #                              [0, 0, 0, 1.]], device=K.device)
    # gl_transform = torch.eye(4, dtype=K.dtype, device=K.device)
    # gl_transform[1, 1] = gl_transform[2, 2] = -1.
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


def load_metadata(images: dict, param_path: Path, lbs_weights_path: Path) -> Metadata:
    url = ""

    # FIXME: igore k1, k2, p1, p2, is this proper?
    h, w = np.array(Image.open(BytesIO(list(images.values())[0].numpy().tobytes()))).shape[:2]

    smplx_model = SMPLX(
        model_path=os.path.join(MODEL_DIR, 'SMPLX_NEUTRAL.npz'),
        use_pca=False,
        num_pca_comps=12,
        num_betas=10,
        flat_hand_mean=False)

    param = np.load(param_path, allow_pickle=True).item()
    smpl_params = param['smpl_params'].reshape(1, -1)
    scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(
        smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)

    with torch.no_grad():
        output = smplx_model(
            global_orient=torch.zeros_like(global_orient),
            body_pose=torch.zeros_like(pose),
            betas=betas.to(torch.float32),
            left_hand_pose=torch.zeros_like(left_hand_pose),
            right_hand_pose=torch.zeros_like(right_hand_pose),
            expression=torch.zeros_like(expression),
            jaw_pose=torch.zeros_like(jaw_pose),
            leye_pose=torch.zeros_like(leye_pose),
            reye_pose=torch.zeros_like(reye_pose),
            return_full_pose=True,
        )
        tpose_joints = output.joints.detach().cpu().numpy()[0, :55]
        tpose = output.full_pose.detach()
        tpose_vertices = output.vertices.detach().cpu().numpy()[0]

    # move pelvis root rotation and transl to Rh and Th
    Rh = global_orient.detach().cpu().numpy()[0]
    Th = transl.detach().cpu().numpy()[0]
    Th = Th + tpose_joints[0] - _rvec_to_rmtx(Rh) @ tpose_joints[0]

    with torch.no_grad():
        output = smplx_model(
            body_pose=pose,
            betas=betas,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_full_pose=True,
        )
    joints = output.joints.detach().cpu().numpy()[0, :55]
    full_pose = (output.full_pose - smplx_model.pose_mean).detach().cpu().numpy().reshape(55, 3)
    full_pose[0] = 0.

    camera_params = param['poses']

    timestamps = []
    cameras = []
    poses = [] # global_Rs & global_Ts
    poses_tpose = [] # global_Rs & global_Ts of tpose
    tposes_joints = []
    joints_oris = []
    lbs_weights_imgs = []

    for i in range(len(images)):
        key = f'frame_{i:06d}'

        if camera_params[i][1].shape[0] == 4:
            fx, fy, cx, cy = camera_params[i][1]  # fx, fy, cx, cy
        else:
            fx, fy, cx, cy = 1120, 1120, 320, 448

        extrinsic_params = camera_params[i][0]  # R|T

        # Convert COLMAP coordinates to Pyrender-compatible transformation
        extrinsic_params_inv = torch.inverse(extrinsic_params.clone())
        scale_factor = extrinsic_params_inv[:3, :3].norm(dim=1)
        extrinsic_params_inv[:3, 1:3] = -extrinsic_params_inv[:3, 1:3]
        extrinsic_params_inv[3, :3] = 0

        intrinsic = [fx / w, fy / h, cx / w, cy / h, 0.0, 0.0]
        w2c = extrinsic_params.detach().cpu().numpy()
        w2c = apply_global_tfm_to_camera(
            E=w2c,
            Rh=Rh,
            Th=Th)
        w2c = w2c[:3, :]
        w2c = w2c.flatten()

        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)

        # extract number from string like "images/frame_00002.png"
        timestamps.append(i)

        cnl_gtfms = get_canonical_global_tfms(tpose_joints, use_smplx=True)

        # tpose_joints = tpose_joints
        poses_angles = full_pose

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
            np.zeros_like(full_pose), tpose_joints, use_smplx=True
        )
        global_Rs_Tpose, global_Ts_Tpose = get_global_RTs(
            cnl_gtfms, dst_Rs_Tpose, dst_Ts_Tpose,
            use_smplx=True)
        pose_Tpose = np.concatenate([
            global_Rs_Tpose.reshape(-1),
            global_Ts_Tpose.reshape(-1)
        ])

        poses.append(pose)
        poses_tpose.append(pose_Tpose)
        tposes_joints.append(tpose_joints)
        joints_oris.append(full_pose)

        if RASTERIZE_LBS_WEIGHTS:
            rasterize_context = nvdiffrast.torch.RasterizeCudaContext(device='cuda')
            vertex_obs = apply_lbs_to_means(torch.tensor(tpose_vertices)[None], torch.tensor(global_Rs)[None],
                                            torch.tensor(global_Ts)[None], torch.tensor(smplx_model.lbs_weights)[None])
            vertex_obs = vertex_obs.detach().numpy()[0]
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            intrinsics = K.astype(np.float32)
            lbs_weights_img = rasterize_lbs_weights(
                rasterize_context,
                vertex_obs,
                smplx_model.lbs_weights,
                intrinsics, w2c.reshape(3, 4),
                smplx_model.faces,
                [h, w])
            os.makedirs(lbs_weights_path, exist_ok=True)
            path = lbs_weights_path / f"{i:06d}"
            np.savez_compressed(path, lbs_weights=lbs_weights_img.numpy())
            # lbs_weights_imgs.append(lbs_weights_img)

            if DEBUG:
                j = 55
                color = np.array(sns.color_palette("tab10", n_colors=j))
                color = (lbs_weights_img[..., None].detach().cpu().numpy() * color).sum(-2)
                color = (color * 255).astype(np.uint8)
                vertex_obs = np.concatenate([vertex_obs, np.ones((vertex_obs.shape[0], 1))], axis=1)
                vertex_cam = w2c.reshape(3, 4) @ vertex_obs.T
                vertex_cam = vertex_cam[:3, :].T
                vertex_im = intrinsics.astype(np.float32) @ vertex_cam.T
                vertex_im = (vertex_im[:2] / vertex_im[2:]).T
                image = images[f"{i:06d}"]
                image = np.array(Image.open(BytesIO(image.numpy().tobytes())))
                # for x, y in vertex_im.astype(int):
                #     cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
                cv2.imwrite('temp_huge100k/{}.png'.format(key), (color * 0.5 + image * 0.5)[:, :, [2,1,0]].astype(np.uint8))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    poses = torch.tensor(np.stack(poses), dtype=torch.float32)
    poses_tpose = torch.tensor(np.stack(poses_tpose), dtype=torch.float32)
    tposes_joints = torch.tensor(np.stack(tposes_joints), dtype=torch.float32)
    joints_oris = torch.tensor(np.stack(joints_oris), dtype=torch.float32)
    outputs = {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
        "poses": poses,
        "poses_tpose": poses_tpose,
        "tposes_joints": tposes_joints,
        "joints_ori": joints_oris
    }

    # if RASTERIZE_LBS_WEIGHTS:
    #     lbs_weights_imgs = torch.stack(lbs_weights_imgs)
    #     print(lbs_weights_imgs.shape)
    #     outputs["lbs_weights"] = lbs_weights_imgs

    return outputs, {
        "vertex": tpose_vertices,
        "weights": smplx_model.lbs_weights,
        "faces": smplx_model.faces
    }


def vis_example(example, canonical_infos):
    vertex = canonical_infos['vertex']
    lbs_weights = canonical_infos['weights']
    faces = canonical_infos['faces']

    scene_name = example["key"]

    N = example["timestamps"].shape[0]
    for n in range(N):
        image = np.array(Image.open(BytesIO(example["images"][n].numpy().tobytes())))
        h, w = image.shape[:2]
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
        # vertex_im -= 0.5
        for i, (x, y) in enumerate(vertex_im.astype(int)):
            cv2.circle(image, (x, y), 2, [255, 0, 0], -1)
        print('temp_huge100k/{}_{:04d}.png'.format(scene_name, n), flush=True)
        cv2.imwrite('temp_huge100k/{}_{:04d}.png'.format(scene_name, n), image)

        # if RASTERIZE_LBS_WEIGHTS:
        #     j = 55
        #     color = np.array(sns.color_palette("tab10", n_colors=j))
        #     color = (example["lbs_weights"][n, ..., None].detach().cpu().numpy() * color).sum(-2)
        #     color = (color * 255).astype(np.uint8)
        #     Image.fromarray(color).save('temp_huge100k/{}_{:04d}_lbs.png'.format(scene_name, n))
        #
        #     Image.fromarray((color * 0.5 + image * 0.5).astype(np.uint8)).save('temp_huge100k/{}_{:04d}_overlap.png'.format(scene_name, n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="flux_batch1")
    # parser.add_argument("--subdir", type=str, default="images0")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    subset = args.subset
    # subdir = args.subdir
    start_idx = args.start_idx
    end_idx = args.end_idx

    path = INPUT_DIR

    split_files = {}
    for filename in (INPUT_DIR.parent / "splits").iterdir():
        if filename.name.startswith(subset):
            if "train" in filename.name:
                split_files["train"] = filename.name
            elif "val" in filename.name:
                split_files["val"] = filename.name
            elif "test" in filename.name:
                split_files["test"] = filename.name

    for stage in ("train", "val", "test"):

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} ({chunk_size / 1e6:.2f} MB).", flush=True
            )
            dir = OUTPUT_DIR / subset / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        filelist = np.load(INPUT_DIR.parent / "splits" / split_files[stage], allow_pickle=True)
        keylist = []
        for file in filelist:
            infos = file["video_path"].split("/")
            key = infos[-1][:-4]
            subdir = infos[-3]
            keylist.append((subdir, key))

        for (subdir, key) in tqdm(keylist):
            param_dir = INPUT_DIR / subset / subdir / "param"
            image_dir = INPUT_DIR / subset / subdir / "images"

            if not (INPUT_DIR / subset / subdir / "images" / key).exists():
                continue

            param_path = param_dir / (key + '.npy')
            video_path = INPUT_DIR / subset / subdir / "videos" / (key + '.mp4')

            images = load_images(INPUT_DIR / subset / subdir / "images" / key)
            masks = load_images(INPUT_DIR / subset / subdir / "masks" / key)
            lbs_weights_path = LBS_WEIGHTS_SAVE_DIR / subset / subdir / key
            example, canonical_info = load_metadata(images, param_path, lbs_weights_path)
            example["key"] = subset + '_' + subdir + '_' + key

            image_names = [f"{timestamp.item():0>6}" for timestamp in example["timestamps"]]
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

            # print(len(images), images[0].nbytes)
            num_bytes = get_size(str(video_path).replace('.mp4', '').replace('videos', 'images')) \
                + get_size(str(video_path).replace('.mp4', '').replace('videos', 'masks'))
            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).", flush=True)
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

            if DEBUG:
                vis_example(example, canonical_info)

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / subset / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)
