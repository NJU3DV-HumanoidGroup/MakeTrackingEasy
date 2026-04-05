"""
推理逻辑：SMPL-X AMASS NPZ → G1 DOF/root_trans/root_rot

从 tools/inference.py 提取，路径适配 HF demo 目录结构。
"""
import os
import time
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from copy import deepcopy
from scipy import signal as sp_signal
from smplx import SMPLX

from mmengine.registry import MODELS

from src.utils.rotation_conversions import (
    axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d,
    rotation_6d_to_matrix, matrix_to_quaternion,
)

# 确保 mmengine 注册表加载
import src  # noqa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HF_REPO_ID = "RayZhao/NMR"


def _ensure_large_files():
    """如本地不存在，从 HuggingFace Hub 自动下载大文件。"""
    weight_path = os.path.join(BASE_DIR, 'weights', 'epoch_30.pth')
    if not os.path.exists(weight_path):
        print(f"Downloading model checkpoint from HuggingFace Hub ({HF_REPO_ID})...")
        hf_hub_download(repo_id=HF_REPO_ID, filename="weights/epoch_30.pth",
                        local_dir=BASE_DIR)

    smplx_path = os.path.join(BASE_DIR, 'assets', 'SMPLX_NEUTRAL.npz')
    if not os.path.exists(smplx_path):
        print(f"Downloading SMPL-X model from HuggingFace Hub ({HF_REPO_ID})...")
        hf_hub_download(repo_id=HF_REPO_ID, filename="assets/SMPLX_NEUTRAL.npz",
                        local_dir=BASE_DIR)

# ---------- 模型配置（内联，不依赖外部 .py 文件） ----------

MODEL_CFG = dict(
    init_cfg=None,
    n_embd=512,
    smplx_vqvae_cfg=dict(
        decoder_cfg=dict(
            activation='relu', depth=3, dilation_growth_rate=3, down_t=2,
            input_emb_width=140, norm=None, output_emb_width=512,
            type='DecoderAttn', width=512),
        encoder_cfg=dict(
            activation='relu', depth=3, dilation_growth_rate=3, down_t=2,
            input_emb_width=140, norm=None, output_emb_width=512,
            stride_t=2, type='EncoderAttn', width=512),
        quantizer_cfg=dict(dim=512, levels=[8, 8, 6, 5], type='FSQ'),
        type='VQVAE'),
    transformer_cfg=dict(
        block_size=1024, n_embd=512, n_head=8, n_layer=8,
        type='LLaMAHF_Fwd', vocab_size=512),
    type='RetargetTransformerPredMotion_no_smplvq')


# ---------- 推理常量 ----------

FPS = 30
CHUNK_SECONDS = 4
CHUNK_FRAMES = (FPS * CHUNK_SECONDS // 4) * 4  # 120
OVERLAP_FRAMES = 32
STRIDE_FRAMES = CHUNK_FRAMES - OVERLAP_FRAMES


# ---------- 数据加载 ----------

def load_smpl_data(file_path):
    """从 AMASS NPZ 加载 SMPL-X 运动参数。

    支持字段:
      - transl/global_orient/body_pose (标准 SMPL-X)
      - trans/root_orient/pose_body (AMASS 格式，Z-up)
    """
    data = np.load(file_path, allow_pickle=True)

    if 'transl' in data:
        transl = torch.from_numpy(data['transl']).float()
        global_orient = torch.from_numpy(data['global_orient']).float()
        body_pose = torch.from_numpy(data['body_pose']).float()

        if 'mocap_frame_rate' in data:
            src_fps = float(data['mocap_frame_rate'])
            if src_fps > 30:
                step = int(src_fps / 30)
                transl = transl[::step]
                global_orient = global_orient[::step]
                body_pose = body_pose[::step]

        return transl, global_orient, body_pose, None
    else:
        # AMASS 格式：Z-up → Y-up
        transl = torch.from_numpy(data['trans']).float()
        global_orient = torch.from_numpy(data['root_orient']).float()
        body_pose = torch.from_numpy(data['pose_body']).float()

        rot_zup_to_yup = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).float()
        transl = torch.einsum('ij,tj->ti', rot_zup_to_yup, transl)
        go_mat = axis_angle_to_matrix(global_orient)
        go_mat = torch.einsum('ij,tjk->tik', rot_zup_to_yup, go_mat)
        global_orient = matrix_to_axis_angle(go_mat)

        if 'mocap_frame_rate' in data:
            src_fps = float(data['mocap_frame_rate'])
            if src_fps > 30:
                step = int(src_fps / 30)
                transl = transl[::step]
                global_orient = global_orient[::step]
                body_pose = body_pose[::step]

        return transl, global_orient, body_pose, None


def preprocess_smpl(file_path, smplx_model, betas, device):
    """将 SMPL-X NPZ 转换为 (T, 140) 运动特征向量。"""
    transl, global_orient, body_pose, seq_betas = load_smpl_data(file_path)
    if seq_betas is not None:
        betas = seq_betas

    N = transl.shape[0]
    frame_params = dict(
        transl=transl.to(device),
        global_orient=global_orient.to(device),
        body_pose=body_pose.to(device),
        betas=betas.unsqueeze(0).repeat(N, 1).float().to(device),
        leye_pose=torch.zeros((N, 3), device=device),
        reye_pose=torch.zeros((N, 3), device=device),
        left_hand_pose=torch.zeros((N, 45), device=device),
        right_hand_pose=torch.zeros((N, 45), device=device),
        jaw_pose=torch.zeros((N, 3), device=device),
        expression=torch.zeros((N, 100), device=device),
    )

    with torch.no_grad():
        output = smplx_model(**frame_params)
    position_data = output.joints.detach().cpu()[:, :22]

    global_orient_mat = axis_angle_to_matrix(global_orient)

    position_val_data = position_data[1:] - position_data[:-1]

    root_idx = 0
    y_min = torch.min(position_data[:, :, 1])
    ori = deepcopy(position_data[0, root_idx])
    ori[1] = y_min
    position_data = position_data - ori

    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]
    position_data_cp = deepcopy(position_data)
    position_data[:, :, 0] -= position_data_cp[:, 0:1, 0]
    position_data[:, :, 2] -= position_data_cp[:, 0:1, 2]

    T, njoint, _ = position_data.shape
    final_x = torch.zeros((T, 2 + 6 + njoint * 3 + njoint * 3))
    final_x[1:, 0] = velocities_root[:, 0]
    final_x[1:, 1] = velocities_root[:, 2]
    final_x[:, 2:8] = matrix_to_rotation_6d(global_orient_mat)
    final_x[:, 8:8 + njoint * 3] = position_data.flatten(1, 2)
    final_x[1:, 8 + njoint * 3:8 + njoint * 6] = position_val_data.flatten(1, 2)

    return final_x


def postprocess_g1(pred_motion, apply_filter=True):
    """从 G1 217 维运动向量提取 DOF/root_trans/root_rot。"""
    T = pred_motion.shape[0]
    rot_mat = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float()

    pred_trans = pred_motion[:, 8:8 + 30 * 3].reshape(T, -1, 3)[:, 0]
    pred_trans[:, [0, 2]] += torch.cumsum(pred_motion[:, :2], dim=0)
    pred_trans = torch.einsum('ij,tj->ti', rot_mat, pred_trans)

    pred_rot_mat = rotation_6d_to_matrix(pred_motion[:, 2:8])
    pred_rot_mat = torch.einsum('ij,tjk->tik', rot_mat, pred_rot_mat)
    pred_rot_quat = matrix_to_quaternion(pred_rot_mat)

    pred_dof = pred_motion[:, -29:]

    if apply_filter and T >= 13:
        _b, _a = sp_signal.butter(4, 5 / (30.0 / 2), btype='low')
        pred_trans = torch.from_numpy(
            sp_signal.filtfilt(_b, _a, pred_trans.numpy(), axis=0).copy()
        ).to(pred_trans.dtype)

        pred_rot_quat_np = sp_signal.filtfilt(_b, _a, pred_rot_quat.numpy(), axis=0).copy()
        pred_rot_quat_np /= np.linalg.norm(pred_rot_quat_np, axis=-1, keepdims=True)
        pred_rot_quat = torch.from_numpy(pred_rot_quat_np).to(pred_rot_quat.dtype)

        pred_dof = torch.from_numpy(
            sp_signal.filtfilt(_b, _a, pred_dof.numpy(), axis=0).copy()
        ).to(pred_dof.dtype)

    return pred_dof, pred_rot_quat, pred_trans


# ---------- 旋转规范化工具 ----------

def _make_y_rot(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _extract_yaw(rot_6d):
    R = rotation_6d_to_matrix(rot_6d.unsqueeze(0))[0]
    forward = R[:, 2]
    return torch.atan2(forward[0], forward[2])


def _rotate_motion_features(motion, R, n_joints, rotate_6d=True):
    result = motion.clone()
    vx, vz = motion[:, 0], motion[:, 1]
    result[:, 0] = R[0, 0] * vx + R[0, 2] * vz
    result[:, 1] = R[2, 0] * vx + R[2, 2] * vz
    if rotate_6d:
        rot_mat = rotation_6d_to_matrix(motion[:, 2:8])
        rot_mat = torch.einsum('ij,tjk->tik', R, rot_mat)
        result[:, 2:8] = matrix_to_rotation_6d(rot_mat)
    pos_start, pos_end = 8, 8 + n_joints * 3
    pos = motion[:, pos_start:pos_end].reshape(-1, n_joints, 3)
    pos = torch.einsum('ij,tnj->tni', R, pos)
    result[:, pos_start:pos_end] = pos.reshape(-1, n_joints * 3)
    vel_start, vel_end = pos_end, pos_end + n_joints * 3
    if vel_end <= motion.shape[1]:
        vel = motion[:, vel_start:vel_end].reshape(-1, n_joints, 3)
        vel = torch.einsum('ij,tnj->tni', R, vel)
        result[:, vel_start:vel_end] = vel.reshape(-1, n_joints * 3)
    return result


# ---------- 推理核心 ----------

def _infer_chunk(smplx_motion, model, smplx_mean, smplx_std, g1_mean, g1_std, device):
    yaw = _extract_yaw(smplx_motion[0, 2:8])
    R_canon = _make_y_rot(-yaw)
    R_restore = _make_y_rot(yaw)

    smplx_motion = _rotate_motion_features(smplx_motion, R_canon, n_joints=22)

    smplx_motion = (smplx_motion - smplx_mean) / smplx_std
    smplx_input = smplx_motion.unsqueeze(0).float().to(device)
    motion_length = torch.tensor([smplx_motion.shape[0]]).to(device)

    with torch.no_grad():
        pred_motions, _ = model(smplx_motion=smplx_input, motion_length=motion_length, mode='predict')

    pred_motion = pred_motions[0].cpu()
    pred_motion = pred_motion * g1_std + g1_mean

    pred_motion = _rotate_motion_features(pred_motion, R_restore, n_joints=30)
    return pred_motion


def infer_single(file_path, model, smplx_model, betas,
                 smplx_mean, smplx_std, g1_mean, g1_std,
                 device, apply_filter=True):
    t0 = time.time()
    smplx_motion = preprocess_smpl(file_path, smplx_model, betas, device)

    T_orig = smplx_motion.shape[0]
    if T_orig < 4:
        return None, None
    T_pad = ((T_orig + 3) // 4) * 4
    if T_pad > T_orig:
        pad = smplx_motion[-1:].repeat(T_pad - T_orig, 1)
        smplx_motion = torch.cat([smplx_motion, pad], dim=0)
    T = T_pad
    t_preprocess = time.time() - t0

    t1 = time.time()
    if T <= CHUNK_FRAMES:
        pred_motion = _infer_chunk(smplx_motion, model, smplx_mean, smplx_std, g1_mean, g1_std, device)
    else:
        chunks = []
        starts = []
        for start in range(0, T, STRIDE_FRAMES):
            end = min(start + CHUNK_FRAMES, T)
            seg_len = (end - start) // 4 * 4
            if seg_len < 4:
                break
            chunk = smplx_motion[start:start + seg_len]
            chunks.append(_infer_chunk(chunk, model, smplx_mean, smplx_std, g1_mean, g1_std, device))
            starts.append(start)

        pred_motion = chunks[0]
        for i in range(1, len(chunks)):
            overlap = starts[i - 1] + len(chunks[i - 1]) - starts[i]
            if overlap > 0:
                w = torch.linspace(0, 1, overlap).unsqueeze(1)
                prev_tail = pred_motion[-overlap:]
                curr_head = chunks[i][:overlap]
                blended = prev_tail * (1 - w) + curr_head * w
                pred_motion = torch.cat([pred_motion[:-overlap], blended, chunks[i][overlap:]], dim=0)
            else:
                pred_motion = torch.cat([pred_motion, chunks[i]], dim=0)
    t_infer = time.time() - t1

    pred_motion = pred_motion[:T_orig]
    t2 = time.time()
    pred_dof, pred_rot_quat, pred_trans = postprocess_g1(pred_motion, apply_filter=apply_filter)
    t_postprocess = time.time() - t2

    t_total = time.time() - t0
    timing = dict(preprocess=t_preprocess, infer=t_infer, postprocess=t_postprocess, total=t_total)

    return dict(
        dof=pred_dof.numpy(),
        root_trans=pred_trans.numpy(),
        root_rot_quat=pred_rot_quat.numpy(),
        source_path=file_path,
    ), timing


# ---------- 模型加载 ----------

def load_all(weights_dir=None, assets_dir=None, device=None):
    """加载模型、SMPLX 体模型和标准化参数。首次运行会自动从 HuggingFace Hub 下载大文件。"""
    _ensure_large_files()

    if weights_dir is None:
        weights_dir = os.path.join(BASE_DIR, 'weights')
    if assets_dir is None:
        assets_dir = os.path.join(BASE_DIR, 'assets')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PyTorch 2.6 兼容
    _torch_load = torch.load
    torch.load = lambda *args, **kwargs: _torch_load(*args, weights_only=kwargs.pop('weights_only', False), **kwargs)

    # 构建并加载模型
    model = MODELS.build(MODEL_CFG)
    ckpt = torch.load(os.path.join(weights_dir, 'epoch_30.pth'), map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)

    # 加载 SMPLX 体模型
    smplx_model = SMPLX(
        model_path=os.path.join(assets_dir, 'SMPLX_NEUTRAL.npz'),
        use_pca=False, num_expression_coeffs=100, num_betas=10, ext='npz'
    ).to(device).eval()

    betas = torch.from_numpy(np.load(os.path.join(assets_dir, 'betas.npy'))).float()

    # 标准化参数
    smplx_mean = torch.from_numpy(np.load(os.path.join(weights_dir, 'smplx_mean.npy'))).float()
    smplx_std = torch.from_numpy(np.load(os.path.join(weights_dir, 'smplx_std.npy'))).float()
    g1_mean = torch.from_numpy(np.load(os.path.join(weights_dir, 'gmr_mean.npy'))).float()
    g1_std = torch.from_numpy(np.load(os.path.join(weights_dir, 'gmr_std.npy'))).float()

    return model, smplx_model, betas, smplx_mean, smplx_std, g1_mean, g1_std, device
