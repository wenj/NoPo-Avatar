'''
MIT License

Copyright (c) 2024 Youngjoong Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import sys
import math
import trimesh
import os
import cv2


def find_3d_vertices_for_uv(faces):
    uv_to_vertices = {}

    vertices = []
    uvs = []

    for face in faces:
        for idx in range(3):

            vertex_index, uv_index = face[idx]

            if uv_index in uv_to_vertices:
                uv_to_vertices[uv_index].add(vertex_index)
            else:
                uv_to_vertices[uv_index] = set()
                uv_to_vertices[uv_index].add(vertex_index)

    return uv_to_vertices


def load_obj(file_path):

    vertices = []
    faces = []
    uvs = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == 'v':

                x, y, z = map(float, tokens[1:4])
                vertices.append((x, y, z))

            elif tokens[0] == 'vt':

                u, v = map(float, tokens[1:3])
                uvs.append((u, v))

            elif tokens[0] == 'f':

                face = []
                for token in tokens[1:]:
                    vertex_info = token.split('/')
                    vertex_index = int(vertex_info[0]) - 1
                    uv_index = int(vertex_info[1]) - 1 if len(
                        vertex_info) > 1 else None
                    face.append((vertex_index, uv_index))
                faces.append(face)

    return vertices, faces, uvs


use_smplx = True
da_pose = False

if use_smplx:
    from smplx import SMPLX
    smplx_model = SMPLX(model_path='datasets/smplx', gender='male')
    lbs_weights = smplx_model.lbs_weights.to(dtype=torch.float32, device='cuda')
else:
    from smplx import SMPL
    smpl_model = SMPL(model_path='datasets/smplx', gender='neutral')
    lbs_weights = smpl_model.lbs_weights.to(dtype=torch.float32, device='cuda')

for resolution in [256, 512, 1024]:
    glctx = dr.RasterizeCudaContext()

    if use_smplx:
        smplx_fp = "assets/templates/smplx_uv/smplx_uv.obj"
    else:
        smplx_fp = "assets/templates/smpl_uv/smpl_uv.obj"

    vertices_tpose, faces, uvs = load_obj(smplx_fp)
    vertices_tpose = np.array(vertices_tpose)
    if da_pose:
        t_pose = smplx_model.body_pose.detach()
        t_pose.requires_grad = False
        t_pose[0, 2] = 1.0
        t_pose[0, 5] = -1.0
        vertices_tpose = smplx_model(pose=t_pose).vertices[0].detach().cpu().numpy()

    faces = np.array(faces)
    uvs = np.array(uvs)
    uv_pts_mapping = find_3d_vertices_for_uv(faces)

    n_uvs = uvs.shape[0]

    pos = uvs
    pos = 2 * pos - 1
    final_pos = np.stack(
        [pos[..., 0], pos[..., 1], np.zeros_like(pos[..., 0]),
         np.ones_like(pos[..., 0])], axis=-1)
    final_pos = final_pos.reshape((1, -1, 4))

    pos_uv = torch.from_numpy(final_pos).to(dtype=torch.float32, device='cuda')
    tri_uv = torch.from_numpy(faces[...,1]).to(dtype=torch.int32, device='cuda')
    rast_uv_space, _ = dr.rasterize(glctx, pos_uv, tri_uv, resolution=[resolution, resolution])

    face_id_raw = rast_uv_space[..., 3:]
    face_id = face_id_raw[0]

    vertices = torch.from_numpy(vertices_tpose).to(dtype=torch.float32, device='cuda')
    attr = []
    for uv_idx in range(len(uvs)):
        for vertex_idx in uv_pts_mapping[uv_idx]:
            attr.append(torch.cat([vertices[vertex_idx], lbs_weights[vertex_idx]], dim=-1))

    attr = torch.stack(attr, dim=0)
    attr = attr[None]

    out, _ = dr.interpolate(attr, rast_uv_space, tri_uv)
    out = out[0].detach().cpu().numpy()

    lbs_weights_rendered = out[..., 3:]
    xyz = out[..., :3]
    mask = lbs_weights_rendered.sum(-1) > 0

    # print(mask.shape, lbs_weights.shape, xyz.shape)

    suffix = "_da" if da_pose else ""
    if use_smplx:
        np.save(f'assets/templates/mask_res{resolution}{suffix}.npy', mask)
        np.save(f'assets/templates/lbs_weights_res{resolution}{suffix}.npy', lbs_weights_rendered)
        np.save(f'assets/templates/xyz_res{resolution}{suffix}.npy', xyz)
    else:
        np.save(f'assets/templates/smpl_mask_res{resolution}{suffix}.npy', mask)
        np.save(f'assets/templates/smpl_lbs_weights_res{resolution}{suffix}.npy', lbs_weights_rendered)
        np.save(f'assets/templates/smpl_xyz_res{resolution}{suffix}.npy', xyz)
