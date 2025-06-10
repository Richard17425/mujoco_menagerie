import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------
# UR10e HFMC 示例（强化姿态控制）：任务空间 PD + 伪逆映射 +
# 重力补偿 + 姿态约束（末端 Z 轴始终指向世界坐标系负向）
#   —— 这里对姿态控制部分采用伪逆 + 惯性映射，而非 J_rot.T 直接映射
# ---------------------------------------------------------------

# 1. 加载 MuJoCo 模型并初始化
model_dir = Path("universal_robots_ur10e")
model_xml = model_dir / "scene_Door.xml"

model = mujoco.MjModel.from_xml_path(str(model_xml))
data  = mujoco.MjData(model)

# 如果 XML 中定义了名为 "home" 的关键帧，则重置到该姿态；否则跳过
try:
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
except Exception:
    pass

# 第一次前向计算，更新 qpos, qvel, sensordata, qfrc_bias
mujoco.mj_forward(model, data)

# 启动被动可视化，我们在循环中手动调用 mj_step
viewer = mujoco.viewer.launch_passive(model, data)

# 2. 获取传感器与 Site 索引
force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "eef_force")
site_id         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "attachment_site")

# 3. 控制器增益初始化
# 3.1 任务空间 PD（位置）
Kp_nu     = np.diag([8.0,  8.0,  8.0])    # 位置比例增益 (N/m)
Kd_nu     = np.diag([15.0, 15.0, 15.0])   # 速度阻尼增益 (N·s/m)

# 3.2 力控制环（如果需要调接触力，否则可保持零）
Kp_lambda = np.diag([0.3,  0.3,  0.3])    # 力比例增益 (N/N)
Kd_lambda = np.diag([0.05, 0.05, 0.05])   # 力导数阻尼增益 (N·s/N)

# 3.3 姿态（方向）PD 环节：保持末端 Z 轴朝世界负 Z 方向
Kp_r = np.diag([5.0, 5.0, 5.0])     # 姿态比例增益 (rad)
Kd_r = np.diag([2.0, 2.0, 2.0])     # 姿态阻尼增益 (rad/s)

dt = model.opt.timestep  # MuJoCo 时间步长 (一般 0.002 或 0.001)

# 4. 定义点 A、点 B
p0       = data.site_xpos[site_id].copy()
point_A  = p0.copy()
point_B  = p0 + np.array([0.10, 0.10, 0.00])  # 往 X, Y 各移动 0.1 m

# 5. 力控制期望：自由空间时保持零
f_desired = np.zeros(3)

# 6. 日志容器（供仿真结束后绘图）
pos_log        = []
err_log        = []
force_log      = []
ferr_log       = []
orient_err_log = []  # 记录姿态误差（rad）

# 7. 上一步的力传感器读数（用于计算力导数）
prev_force_meas = np.zeros(3)

# 8. 主 HFMC 控制循环
num_steps  = 2000
half_steps = num_steps // 2

for step in range(num_steps):
    # 8.1 前向计算，更新 qpos, qvel, sensordata, qfrc_bias
    mujoco.mj_forward(model, data)

    # 8.2 当前末端 TCP 位置 (world frame)
    p = data.site_xpos[site_id].copy()

    # 8.3 读取力传感器输出 (world frame)
    f_meas = data.sensordata[force_sensor_id : force_sensor_id + 3].copy()

    # 8.4 平滑插值得到期望位置 (A→B)
    if step < half_steps:
        desired_pos = point_A
    else:
        t = float(step - half_steps) / float(half_steps)  # t ∈ [0, 1]
        desired_pos = (1.0 - t) * point_A + t * point_B

    # 8.5 计算位置误差
    nu_error = desired_pos - p

    # 8.6 计算末端雅可比 J_pos（平动部分）和 J_rot（旋转部分）
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J_pos, J_rot, site_id)

    # 8.7 末端线速度 (world frame)
    raw_v_ee = J_pos @ data.qvel

    # 8.8 任务空间 PD (运动)
    alpha_nu = Kp_nu @ nu_error - Kd_nu @ raw_v_ee

    # 8.9 力误差 ferr = f_desired - f_meas
    ferr = f_desired - f_meas
    # 8.10 力导数 f_dot = (f_meas - prev_force_meas)/dt
    f_dot = (f_meas - prev_force_meas) / dt
    prev_force_meas = f_meas.copy()

    # 8.11 力控制: f_lambda = Kp_lambda · ferr - Kd_lambda · f_dot
    f_lambda = Kp_lambda @ ferr - Kd_lambda @ f_dot

    # 8.12 姿态控制：保持末端 Z 轴朝世界负 Z 方向
    #     使用 data.site_xmat[site_id] 直接读取 3×3 旋转矩阵
    R_flat = data.site_xmat[site_id]       # (9,) 列表形式的旋转矩阵
    R = R_flat.reshape((3, 3))             # 重塑为 3×3
    z_current = R[:, 2]                    # 当前末端 Z 轴方向向量

    # 期望方向：世界坐标系下 “向下” (0, 0, -1)
    z_desired = np.array([0.0, 0.0, -1.0])

    # 计算方向误差向量：cross(z_current, z_desired)
    # 两向量归一化后，cross 的结果近似于旋转轴 × sin(theta)
    z_current_norm = z_current / (np.linalg.norm(z_current) + 1e-9)
    z_desired_norm = z_desired / (np.linalg.norm(z_desired) + 1e-9)
    orient_err_vec = np.cross(z_current_norm, z_desired_norm)
    # 记录姿态误差大小 (rad)
    orient_err_log.append(np.linalg.norm(orient_err_vec))

    # 8.13 计算末端角速度 (world frame)：w_ee = J_rot @ qvel
    w_ee = J_rot @ data.qvel  # (3,) 角速度

    # 8.14 姿态 PD 控制：alpha_r = Kp_r · orient_err_vec - Kd_r · w_ee
    alpha_r = Kp_r @ orient_err_vec - Kd_r @ w_ee  # (3,) 角加速度指令

    # 8.15 组合运动/力/姿态控制指令 (线性与角度拆分处理)
    Sv = np.eye(3)
    Sf = np.zeros((3, 3))
    Sr = np.eye(3)  # 姿态所有轴都参与

    α_trans = Sv @ alpha_nu    # (3,) 线性加速度指令
    α_force = Sf @ f_lambda    # (3,) 力控制部分 (自由空间为 0)
    α_rot   = Sr @ alpha_r     # (3,) 角加速度指令

    # 8.16 重力与偏置补偿：data.qfrc_bias[:nv] = C(q, q̇) + G(q)
    gravity_comp = data.qfrc_bias[: model.nu].copy()  # (nv,)

    # 8.17 计算质量矩阵 M (nv×nv)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)

    # 8.18 伪逆映射：J_pos^+ 用于线性部分
    J_pinv = np.linalg.pinv(J_pos)  # (nv×3)

    # 8.19 计算期望关节加速度：qdd_trans = J_pinv · α_trans
    qdd_trans = J_pinv @ α_trans  # (nv,)

    # 8.20 线性惯性扭矩：τ_trans = M · qdd_trans
    τ_trans = M @ qdd_trans  # (nv,)

    # 8.21 姿态伪逆映射：J_rot^+ 用于角度部分
    J_rot_pinv = np.linalg.pinv(J_rot)  # (nv×3)

    # 8.22 计算期望关节角加速度：qdd_rot = J_rot_pinv · α_rot
    qdd_rot = J_rot_pinv @ α_rot  # (nv,)

    # 8.23 姿态惯性扭矩：τ_orient = M · qdd_rot
    τ_orient = M @ qdd_rot  # (nv,)

    # 8.24 合成最终关节扭矩：τ_task = τ_trans + τ_orient
    τ_task = τ_trans + τ_orient

    # 8.25 最终控制输入：u_joint = τ_task + gravity_comp
    u_joint = τ_task + gravity_comp

    # 8.26 饱和：根据 actuator ctrlrange 设置 (此处设 ±50 N·m)
    ctrl_clipped     = np.clip(u_joint, -50.0, 50.0)
    data.ctrl[:] = ctrl_clipped[: model.nu]

    # 8.27 推进仿真并可视化
    mujoco.mj_step(model, data)
    viewer.sync()

    # 8.28 记录日志
    pos_log.append(p.copy())
    err_log.append(nu_error.copy())
    force_log.append(f_meas.copy())
    ferr_log.append(ferr.copy())

    # 8.29 每隔 200 步打印一次状态
    if step % 200 == 0:
        pos_err_norm   = np.linalg.norm(nu_error)
        f_err_norm     = np.linalg.norm(ferr)
        orient_err_mag = np.linalg.norm(orient_err_vec)
        phase = "Holding A" if step < half_steps else "Moving to B"
        print(
            f"[Step {step:4d}] Phase→{phase}, "
            f"PosErr={pos_err_norm:.4f} m, "
            f"ForceErr={f_err_norm:.4f} N, "
            f"OrientErr={orient_err_mag:.4f} rad"
        )

# 9. 仿真结束后：绘制 3D 轨迹 & 误差

pos_array      = np.array(pos_log)      # shape: (num_steps, 3)
err_array      = np.array(err_log)      # shape: (num_steps, 3)
force_array    = np.array(force_log)    # shape: (num_steps, 3)
ferr_array     = np.array(ferr_log)     # shape: (num_steps, 3)
orient_err_arr = np.array(orient_err_log)  # shape: (num_steps,)

# 9.1 绘制 3D 末端轨迹
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(
    pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
    label='End-Effector Path', linewidth=2
)
ax.scatter(
    pos_array[0, 0], pos_array[0, 1], pos_array[0, 2],
    color='green', s=50, label='Start (A)'
)
ax.scatter(
    pos_array[-1, 0], pos_array[-1, 1], pos_array[-1, 2],
    color='red', s=50, label='End (B)'
)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D End-Effector Trajectory (HFMC + Attitude Constraint)')
ax.legend()
plt.tight_layout()

plt.show()
