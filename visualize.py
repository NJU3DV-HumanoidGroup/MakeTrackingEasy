"""
Plotly 3D 骨骼动画可视化：使用 pinocchio FK 将 DOF → 关节位置 → 3D stick figure。
"""
import os
import torch
import numpy as np
import pinocchio as pin
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 从 humanoid_metric.py 复制的映射表
JOINT_MAPPING = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
BODY_MAPPING = [4, 20, 34, 6, 22, 36, 8, 24, 38, 10, 26, 46, 62, 12, 28, 48, 64, 14, 30, 50, 66, 52, 68, 54, 70, 56, 72, 58, 74]

# 骨骼连接 (body_mapping 索引对)
# body_mapping 索引含义:
#  0:L_hip_pitch  1:R_hip_pitch  2:waist_yaw  3:L_hip_roll  4:R_hip_roll
#  5:waist_roll   6:L_hip_yaw    7:R_hip_yaw   8:torso       9:L_knee
# 10:R_knee      11:L_shoulder_pitch 12:R_shoulder_pitch  13:L_ankle_pitch
# 14:R_ankle_pitch  15:L_shoulder_roll  16:R_shoulder_roll
# 17:L_ankle_roll  18:R_ankle_roll  19:L_shoulder_yaw  20:R_shoulder_yaw
# 21:L_elbow  22:R_elbow  23:L_wrist_roll  24:R_wrist_roll
# 25:L_wrist_pitch 26:R_wrist_pitch  27:L_wrist_yaw  28:R_wrist_yaw
BONES = [
    # Hip connection
    (0, 1),
    # Spine: hips → waist → torso
    (0, 2), (1, 2), (2, 5), (5, 8),
    # Left leg: hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll
    (0, 3), (3, 6), (6, 9), (9, 13), (13, 17),
    # Right leg
    (1, 4), (4, 7), (7, 10), (10, 14), (14, 18),
    # Left arm: torso → shoulder_pitch → shoulder_roll → shoulder_yaw → elbow → wrist...
    (8, 11), (11, 15), (15, 19), (19, 21), (21, 23), (23, 25), (25, 27),
    # Right arm
    (8, 12), (12, 16), (16, 20), (20, 22), (22, 24), (24, 26), (26, 28),
]

# 关节颜色分组
JOINT_COLORS = {}
for i in [0, 3, 6, 9, 13, 17]:
    JOINT_COLORS[i] = 'blue'       # 左腿
for i in [1, 4, 7, 10, 14, 18]:
    JOINT_COLORS[i] = 'red'        # 右腿
for i in [2, 5, 8]:
    JOINT_COLORS[i] = 'green'      # 躯干
for i in [11, 15, 19, 21, 23, 25, 27]:
    JOINT_COLORS[i] = 'dodgerblue' # 左臂
for i in [12, 16, 20, 22, 24, 26, 28]:
    JOINT_COLORS[i] = 'orangered'  # 右臂

BONE_COLORS = {
    'left_leg': 'blue', 'right_leg': 'red', 'spine': 'green',
    'left_arm': 'dodgerblue', 'right_arm': 'orangered',
}


def _load_pin_model(urdf_path=None):
    if urdf_path is None:
        urdf_path = os.path.join(BASE_DIR, 'assets', 'g1_29dof.urdf')
    return pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())


def compute_joint_positions(dof, rot_quat, transl, pin_model=None):
    """DOF + root pose → 关节世界坐标位置。

    Args:
        dof: (T, 29) 关节角度
        rot_quat: (T, 4) wxyz 四元数
        transl: (T, 3) 根位置
        pin_model: pinocchio model (可选，传入避免重复加载)

    Returns:
        (T, 29, 3) 关节位置
    """
    if pin_model is None:
        pin_model = _load_pin_model()
    data = pin_model.createData()

    if isinstance(dof, torch.Tensor):
        dof = dof.numpy()
    if isinstance(rot_quat, torch.Tensor):
        rot_quat = rot_quat.numpy()
    if isinstance(transl, torch.Tensor):
        transl = transl.numpy()

    T = dof.shape[0]
    mapped_joint = np.zeros((T, len(JOINT_MAPPING)))
    for i, jm in enumerate(JOINT_MAPPING):
        mapped_joint[:, jm] = dof[:, i]

    joint_positions = np.zeros((T, len(BODY_MAPPING), 3))
    for t in range(T):
        q = np.zeros(pin_model.nq)
        q[0:3] = transl[t]
        q[3:7] = rot_quat[t, [1, 2, 3, 0]]  # wxyz → xyzw
        q[7:] = mapped_joint[t]

        pin.forwardKinematics(pin_model, data, q)
        pin.updateFramePlacements(pin_model, data)

        for lab_idx, pino_idx in enumerate(BODY_MAPPING):
            joint_positions[t, lab_idx] = data.oMf[pino_idx].translation

    return joint_positions


def create_skeleton_animation(dof, rot_quat, transl, fps=30, max_display_fps=10):
    """创建 Plotly 3D 骨骼动画。

    Args:
        dof: (T, 29)
        rot_quat: (T, 4) wxyz
        transl: (T, 3)
        fps: 原始帧率
        max_display_fps: 显示帧率（降采样）

    Returns:
        plotly Figure
    """
    pin_model = _load_pin_model()
    joint_pos = compute_joint_positions(dof, rot_quat, transl, pin_model)
    T = joint_pos.shape[0]

    # 降采样用于显示
    step = max(1, fps // max_display_fps)
    display_indices = list(range(0, T, step))
    if display_indices[-1] != T - 1:
        display_indices.append(T - 1)

    # 计算坐标范围
    all_pos = joint_pos[display_indices]
    margin = 0.3
    x_range = [all_pos[:, :, 0].min() - margin, all_pos[:, :, 0].max() + margin]
    y_range = [all_pos[:, :, 1].min() - margin, all_pos[:, :, 1].max() + margin]
    z_range = [all_pos[:, :, 2].min() - margin, all_pos[:, :, 2].max() + margin]

    # 构建帧
    frames = []
    for frame_idx, t in enumerate(display_indices):
        pos = joint_pos[t]
        traces = []

        # 关节点
        colors = [JOINT_COLORS.get(i, 'gray') for i in range(len(BODY_MAPPING))]
        traces.append(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            marker=dict(size=4, color=colors),
            name='joints',
        ))

        # 骨骼线段
        bone_x, bone_y, bone_z = [], [], []
        for a, b in BONES:
            bone_x.extend([pos[a, 0], pos[b, 0], None])
            bone_y.extend([pos[a, 1], pos[b, 1], None])
            bone_z.extend([pos[a, 2], pos[b, 2], None])
        traces.append(go.Scatter3d(
            x=bone_x, y=bone_y, z=bone_z,
            mode='lines',
            line=dict(width=4, color='dimgray'),
            name='bones',
        ))

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            traces=[0, 1],
        ))

    # 初始帧
    init_data = frames[0].data

    # 构建 Figure
    fig = go.Figure(
        data=list(init_data),
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=x_range, title='X'),
                yaxis=dict(range=y_range, title='Y'),
                zaxis=dict(range=z_range, title='Z'),
                aspectmode='manual',
                aspectratio=dict(
                    x=(x_range[1] - x_range[0]),
                    y=(y_range[1] - y_range[0]),
                    z=(z_range[1] - z_range[0]),
                ),
                camera=dict(
                    eye=dict(x=1.5, y=0.5, z=0.8),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            title=f'G1 Robot Skeleton ({T} frames @ {fps} FPS, showing {len(display_indices)} frames)',
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(
                             frame=dict(duration=1000 // max_display_fps, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=0),
                         )]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(
                             frame=dict(duration=0, redraw=False),
                             mode='immediate',
                             transition=dict(duration=0),
                         )]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[dict(
                    args=[[str(i)], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0),
                    )],
                    label=f'{display_indices[i] / fps:.1f}s',
                    method='animate',
                ) for i in range(len(display_indices))],
                x=0.1, len=0.8,
                xanchor='left',
                y=0, yanchor='top',
                currentvalue=dict(prefix='Time: ', visible=True),
                transition=dict(duration=0),
            )],
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        ),
        frames=frames,
    )

    return fig
