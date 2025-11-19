from math import cos, sin
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.transforms.so3 import so3_exp_map


SMPL_JOINT_IDX = {
    'pelvis_root': 0,
    'left_hip': 1,
    'right_hip': 2,
    'belly_button': 3,
    'left_knee': 4,
    'right_knee': 5,
    'lower_chest': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'upper_chest': 9,
    'left_toe': 10,
    'right_toe': 11,
    'neck': 12,
    'left_clavicle': 13,
    'right_clavicle': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_thumb': 22,
    'right_thumb': 23
}

SMPL_PARENT = {
    0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18,
    21: 19, 22: 20, 23: 21}

SMPL_LEAVES = [key for key in SMPL_PARENT.keys() if key not in SMPL_PARENT.values()]

SMPL_N_JOINTS = 24
SMPL_N_BONES = 23 + len(SMPL_LEAVES)

SMPL_JOINT_FLIP = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22
]

SMPL_BONE_FLIP = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 25, 27, 26]

SMPLX_JOINT_NAMES = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "jaw": 22,
    "left_eye_smplhf": 23,
    "right_eye_smplhf": 24,
    "left_index1": 25,
    "left_index2": 26,
    "left_index3": 27,
    "left_middle1": 28,
    "left_middle2": 29,
    "left_middle3": 30,
    "left_pinky1": 31,
    "left_pinky2": 32,
    "left_pinky3": 33,
    "left_ring1": 34,
    "left_ring2": 35,
    "left_ring3": 36,
    "left_thumb1": 37,
    "left_thumb2": 38,
    "left_thumb3": 39,
    "right_index1": 40,
    "right_index2": 41,
    "right_index3": 42,
    "right_middle1": 43,
    "right_middle2": 44,
    "right_middle3": 45,
    "right_pinky1": 46,
    "right_pinky2": 47,
    "right_pinky3": 48,
    "right_ring1": 49,
    "right_ring2": 50,
    "right_ring3": 51,
    "right_thumb1": 52,
    "right_thumb2": 53,
    "right_thumb3": 54,
}

SMPLX_PARENT = {
    0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
    12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 15, 23: 15, 24: 15, 25: 20, 26: 25, 27: 26, 28: 20, 29: 28, 30: 29, 31: 20,
    32: 31, 33: 32, 34: 20, 35: 34, 36: 35, 37: 20, 38: 37, 39: 38, 40: 21, 41: 40,
    42: 41, 43: 21, 44: 43, 45: 44, 46: 21, 47: 46, 48: 47, 49: 21, 50: 49, 51: 50,
    52: 21, 53: 52, 54: 53}

SMPLX_LEAVES = [key for key in SMPLX_PARENT.keys() if key not in SMPLX_PARENT.values()]

SMPLX_N_JOINTS = 55
SMPLX_N_BONES = 54 + len(SMPLX_LEAVES)

SMPLX_JOINT_FLIP = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 22, 24, 23,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
]

SMPLX_BONE_FLIP = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19, 21, 23, 22, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 55, 54, 56, 58, 57, 64, 65, 66, 67, 68, 59, 60, 61, 62, 63]


def _construct_G(R_mtx, T):
    r""" Build 4x4 [R|T] matrix from rotation matrix, and translation vector

    Args:
        - R_mtx: Array (3, 3)
        - T: Array (3,)

    Returns:
        - Array (4, 4)
    """

    G = np.array(
        [[R_mtx[0, 0], R_mtx[0, 1], R_mtx[0, 2], T[0]],
         [R_mtx[1, 0], R_mtx[1, 1], R_mtx[1, 2], T[1]],
         [R_mtx[2, 0], R_mtx[2, 1], R_mtx[2, 2], T[2]],
         [0., 0., 0., 1.]],
        dtype='float32')

    return G


def _construct_G_tensor(R_mtx, T):
    r''' Tile ration matrix and translation vector to build a 4x4 matrix.

	Args:
		R_mtx: Tensor (B, TOTAL_BONES, 3, 3)
		T:     Tensor (B, TOTAL_BONES, 3)

	Returns:
		G:     Tensor (B, TOTAL_BONES, 4, 4)
	'''
    G = torch.zeros(size=(4, 4),
                    dtype=R_mtx.dtype, device=R_mtx.device)
    G[:3, :3] = R_mtx
    G[:3, 3] = T
    G[3, 3] = 1.0

    return G


def _to_skew_matrix(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    vx, vy, vz = v.ravel()
    return np.array([[0, -vz, vy],
                    [vz, 0, -vx],
                    [-vy, vx, 0]])


def _to_skew_matrix_tensor(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    new_v = torch.zeros([3, 3], dtype=torch.float32, device=v.device)
    new_v[0, 1] = -v[2]
    new_v[0, 2] = v[1]
    new_v[1, 0] = v[2]
    new_v[1, 2] = -v[0]
    new_v[2, 0] = -v[1]
    new_v[2, 1] = v[0]
    return new_v


def _rvec_to_rmtx(rvec):
    r''' apply Rodriguez Formula on rotate vector (3,)

    Args:
        - rvec: Array (3,)

    Returns:
        - Array (3, 3)
    '''
    rvec = rvec.reshape(3, 1)

    norm = np.linalg.norm(rvec)
    theta = norm
    r = rvec / (norm + 1e-5)

    skew_mtx = _to_skew_matrix(r)

    return cos(theta)*np.eye(3) + \
           sin(theta)*skew_mtx + \
           (1-cos(theta))*r.dot(r.T)


def _rvec_to_rmtx_tensor(rvec):
    r''' apply Rodriguez Formula on rotate vector (3,)

    Args:
        - rvec: Array (3,)

    Returns:
        - Array (3, 3)
    '''
    rvec = rvec.reshape(3, 1)

    norm = torch.linalg.norm(rvec)
    theta = norm
    r = rvec / (norm + 1e-5)

    skew_mtx = _to_skew_matrix_tensor(r)

    return torch.cos(theta) * torch.eye(3, device=rvec.device, dtype=torch.float32) + \
        torch.sin(theta) * skew_mtx + \
        (1 - torch.cos(theta)) * (r @ (r.T))


def body_pose_to_body_RTs(jangles, tpose_joints, use_smplx=False):
    r""" Convert body pose to global rotation matrix R and translation T.

    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    jangles = jangles.reshape(-1, 3)
    total_joints = jangles.shape[0]
    assert tpose_joints.shape[0] == total_joints

    Rs = np.zeros(shape=[total_joints, 3, 3], dtype='float32')
    Rs[0] = _rvec_to_rmtx(jangles[0, :])

    Ts = np.zeros(shape=[total_joints, 3], dtype='float32')
    Ts[0] = tpose_joints[0, :]

    for i in range(1, total_joints):
        Rs[i] = _rvec_to_rmtx(jangles[i, :])
        Ts[i] = tpose_joints[i, :] - tpose_joints[PARENT[i], :]

    return Rs, Ts


def body_pose_to_body_RTs_tensor(jangles_all, tpose_joints_all, use_smplx=False):
    r""" Convert body pose to global rotation matrix R and translation T.

    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    Rs_all, Ts_all = [], []
    for jangles, tpose_joints in zip(jangles_all, tpose_joints_all):
        jangles = jangles.reshape(-1, 3)
        total_joints = jangles.shape[0]
        assert tpose_joints.shape[0] == total_joints


        Ts = torch.zeros([total_joints, 3], dtype=torch.float32, device=jangles.device)
        Ts[0] = tpose_joints[0, :]
        Rs = so3_exp_map(jangles)

        for i in range(1, total_joints):
            Ts[i] = tpose_joints[i, :] - tpose_joints[PARENT[i], :]

        Rs_all.append(Rs)
        Ts_all.append(Ts)
    return torch.stack(Rs_all), torch.stack(Ts_all)


def get_canonical_global_tfms(canonical_joints, use_smplx=False):
    r""" Convert canonical joints to 4x4 global transformation matrix.

    Args:
        - canonical_joints: Array (Total_Joints, 3)

    Returns:
        - Array (Total_Joints, 4, 4)
    """
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    total_bones = canonical_joints.shape[0]

    gtfms = np.zeros(shape=(total_bones, 4, 4), dtype='float32')
    gtfms[0] = _construct_G(np.eye(3), canonical_joints[0, :])

    for i in range(1, total_bones):
        translate = canonical_joints[i, :] - canonical_joints[PARENT[i], :]
        gtfms[i] = gtfms[PARENT[i]].dot(_construct_G(np.eye(3), translate))

    return gtfms


def get_canonical_global_tfms_tensor(canonical_joints_all, use_smplx=False):
    r""" Convert canonical joints to 4x4 global transformation matrix.

    Args:
        - canonical_joints: Array (Total_Joints, 3)

    Returns:
        - Array (Total_Joints, 4, 4)
    """
    gtfms_all = []
    for canonical_joints in canonical_joints_all:
        if not use_smplx:
            PARENT = SMPL_PARENT
        else:
            PARENT = SMPLX_PARENT

        total_bones = canonical_joints.shape[0]

        gtfms = [_construct_G_tensor(torch.eye(3, device=canonical_joints.device), canonical_joints[0, :])]

        for i in range(1, total_bones):
            translate = canonical_joints[i, :] - canonical_joints[PARENT[i], :]
            gtfms.append(gtfms[PARENT[i]] @ _construct_G_tensor(torch.eye(3, device=canonical_joints.device), translate))

        gtfms_all.append(torch.stack(gtfms))
    return torch.stack(gtfms_all)


def get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=False):
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    total_bones = cnl_gtfms.shape[0]
    dst_gtfms = np.zeros_like(cnl_gtfms)

    local_Gs = np.stack([_construct_G(dst_R, dst_T) for dst_R, dst_T in zip(dst_Rs, dst_Ts)])
    dst_gtfms[0, :, :] = local_Gs[0, :, :]

    for i in range(1, total_bones):
        dst_gtfms[i, :, :] = np.matmul(
            dst_gtfms[PARENT[i], :, :].copy(),
            local_Gs[i, :, :])

    dst_gtfms = dst_gtfms.reshape(-1, 4, 4)

    f_mtx = np.matmul(dst_gtfms, np.linalg.inv(cnl_gtfms))
    f_mtx = f_mtx.reshape(total_bones, 4, 4)

    scale_Rs = f_mtx[:, :3, :3]
    Ts = f_mtx[:, :3, 3]

    return scale_Rs, Ts


def get_global_RTs_tensor(cnl_gtfms_all, dst_Rs_all, dst_Ts_all, use_smplx=False):
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    scale_Rs_all, Ts_all = [], []
    for cnl_gtfms, dst_Rs, dst_Ts in zip(cnl_gtfms_all, dst_Rs_all, dst_Ts_all):
        total_bones = cnl_gtfms.shape[0]
        dst_gtfms = torch.zeros_like(cnl_gtfms)

        local_Gs = torch.stack([_construct_G_tensor(dst_R, dst_T) for dst_R, dst_T in zip(dst_Rs, dst_Ts)])
        dst_gtfms[0, :, :] = local_Gs[0, :, :]

        for i in range(1, total_bones):
            dst_gtfms[i, :, :] = torch.matmul(
                dst_gtfms[PARENT[i], :, :].clone(),
                local_Gs[i, :, :])

        dst_gtfms = dst_gtfms.reshape(-1, 4, 4)

        f_mtx = dst_gtfms @ torch.linalg.inv(cnl_gtfms)
        f_mtx = f_mtx.reshape(total_bones, 4, 4)

        scale_Rs = f_mtx[:, :3, :3]
        Ts = f_mtx[:, :3, 3]

        scale_Rs_all.append(scale_Rs)
        Ts_all.append(Ts)
    return torch.stack(scale_Rs_all), torch.stack(Ts_all)


def apply_global_tfm_to_camera(E, Rh, Th, return_global_tfms=False):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )

    Returns:
        - Array (3, 3)
    """

    global_tfms = np.eye(4)  # (4, 4)
    global_rot = cv2.Rodrigues(Rh)[0].T
    global_trans = Th
    global_tfms[:3, :3] = global_rot
    global_tfms[:3, 3] = -global_rot.dot(global_trans)
    if return_global_tfms:
        return E.dot(np.linalg.inv(global_tfms)), global_tfms
    else:
        return E.dot(np.linalg.inv(global_tfms))


def apply_lbs_to_means(mean, global_Rs, global_Ts, lbs_weights):
    N = mean.shape[1]
    B, J, _ = global_Ts.shape
    global_RTs = global_Rs.new_zeros(*global_Ts.shape[:-1], 4, 4)
    global_RTs[..., :3, :3] = global_Rs
    global_RTs[..., :3, 3] = global_Ts
    global_RTs[..., 3, 3] = 1.

    RTs = torch.matmul(lbs_weights, global_RTs.reshape(B, J, -1)).reshape(B, N, 4, 4)
    Rs = RTs[..., :3, :3]
    Ts = RTs[..., :3, 3]

    mean_posed = torch.matmul(Rs, mean.unsqueeze(-1)).squeeze(-1) + Ts
    return mean_posed


def apply_lbs_to_cov(cov, global_Rs, global_Ts, lbs_weights):
    """
    Parameters
    ----------
    xyzs_canonical : B x N x 3
    global_Rs: B x J x 3 x 3
    global_Ts: B x J x 3
    lbs_weights: B x N x J

    Returns
    -------
    B x N x 3 x 3
    """
    # https://stackoverflow.com/questions/52922647/rotate-covariance-matrix

    # cov_new = R cov R^T
    B, _, _, _ = global_Rs.shape
    R_T_ = global_Rs[:, None, :, :, :].permute(0, 1, 2, 4, 3) # B x 1 x J x 3 x 3
    cov_ = cov[:, :, None, :, :] # B x N x 1 x 3 x 3
    R_ = global_Rs[:, None, :, :, :] # B x 1 x J x 3 x 3
    cov_trans = R_ @ cov_ @ R_T_ # B x N x J x 3 x 3
    cov_new = torch.sum(cov_trans * (lbs_weights[:, :, :, None, None]) ** 2, dim=2) # B x N x 3 x 3
    return cov_new


def apply_lbs_to_gaussians(mean, cov, global_Rs, global_Ts, lbs_weights):
    N = mean.shape[1]
    B, J, _ = global_Ts.shape
    global_RTs = global_Rs.new_zeros(*global_Ts.shape[:-1], 4, 4)
    global_RTs[..., :3, :3] = global_Rs
    global_RTs[..., :3, 3] = global_Ts
    global_RTs[..., 3, 3] = 1.

    RTs = torch.matmul(lbs_weights, global_RTs.reshape(B, J, -1)).reshape(B, N, 4, 4)
    Rs = RTs[..., :3, :3]
    Ts = RTs[..., :3, 3]

    mean_posed = torch.matmul(Rs, mean.unsqueeze(-1)).squeeze(-1) + Ts
    cov_posed = Rs @ cov @ Rs.transpose(-1, -2)

    return mean_posed, cov_posed

    # return apply_lbs(mean, global_Rs, global_Ts, lbs_weights), apply_lbs_to_cov(cov, global_Rs, global_Ts, lbs_weights)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / torch.linalg.norm(vec1, dim=-1, keepdim=True)), (vec2 / torch.linalg.norm(vec2, dim=-1, keepdim=True))
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    s = torch.linalg.norm(v, dim=-1)
    kmat = vec1.new_zeros(a.shape[0], 3, 3)
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    # kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3, device=vec1.device) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))[:, None, None]
    return rotation_matrix


def get_canonical_tfms(tpose_src, tpose_tgt, use_smplx=False):
    # tpose0, tpose1: J x 3
    if use_smplx:
        PARENT = SMPLX_PARENT
        LEAVES = SMPLX_LEAVES
    else:
        PARENT = SMPL_PARENT
        LEAVES = SMPL_LEAVES

    childs = list(PARENT.keys())[1:]
    parents = list(PARENT.values())[1:]

    edge_tgt = tpose_tgt[childs] - tpose_tgt[parents]
    edge_src = tpose_src[childs] - tpose_src[parents]
    scale = (torch.linalg.norm(edge_tgt, dim=-1, keepdim=True) / torch.linalg.norm(edge_src, dim=-1, keepdim=True)).unsqueeze(-1)
    rot = rotation_matrix_from_vectors(edge_src, edge_tgt)
    trans = tpose_tgt[parents] - (scale * rot @ tpose_src[parents].unsqueeze(-1)).squeeze(-1)

    tfms = torch.eye(4, device=tpose_src.device)[None].repeat(len(childs) + len(LEAVES), 1, 1)
    tfms[:len(childs), :3, :3] = scale * rot
    tfms[:len(childs), :3, 3] = trans

    # for i, (child, parent) in enumerate(PARENT.items()):
    #     if parent == -1:
    #         continue
    #
    #     print((tfms[i - 1, :3, :3] @ tpose_src[child].unsqueeze(-1)).squeeze(-1) + tfms[i - 1, :3, 3], tpose_tgt[child])

    return tfms[..., :3, :3], tfms[..., :3, 3]


def bone_lbs_weights_to_joint_lbs_weights(bone_lbs_weights, use_smplx=False):
    if use_smplx:
        n_joints = 55
        PARENT = SMPLX_PARENT
        LEAVES = SMPLX_LEAVES
    else:
        n_joints = 24
        PARENT = SMPL_PARENT
        LEAVES = SMPL_LEAVES

    bones_lbs_weights_normalized = bone_lbs_weights.softmax(-1)
    joint_lbs_weights = bone_lbs_weights.new_zeros(*bone_lbs_weights.shape[:-1], n_joints)
    for i, (child, parent) in enumerate(PARENT.items()):
        if parent == -1:
            continue
        joint_lbs_weights[..., parent] += bones_lbs_weights_normalized[..., i - 1]

    n_edge = len(PARENT.items()) - 1
    for i, parent in enumerate(LEAVES):
        joint_lbs_weights[..., parent] += bones_lbs_weights_normalized[..., i + n_edge]

    joint_lbs_weights = F.normalize(joint_lbs_weights, dim=-1).clamp(min=1e-6)
    return joint_lbs_weights.log()
