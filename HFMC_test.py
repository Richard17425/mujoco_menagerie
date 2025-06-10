import numpy as np
from pathlib import Path
from tqdm import tqdm
import mujoco
import mujoco.viewer
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ------------------------ Load Model ------------------------
model_dir = Path("universal_robots_ur10e")
model_xml = model_dir / "scene_Door.xml"
model = mujoco.MjModel.from_xml_path(str(model_xml))
data = mujoco.MjData(model)

# ------------------------ Trajectory ------------------------
trajectory_points = np.loadtxt("Traj1.csv", delimiter=',')
yaw = - 90 - trajectory_points[:, 3] * 180.0 / np.pi # Convert degrees to radians
traj = trajectory_points[:190, :3] * 3 / 100.0
traj[:, 1] += 1.0
traj[:, 2] += 0.43
traj = gaussian_filter1d(traj, sigma=2, axis=0)

total_frames = len(traj)

# ------------------------ Viewer ------------------------
viewer = mujoco.viewer.launch_passive(model, data)

# ------------------------ Gains ------------------------
Kp = np.diag([1.0] * model.nv)
Kd = np.diag([0.0] * model.nv)

# ------------------------ End-effector site ID ------------------------
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

# ------------------------ Desired constant orientation (Z down) ------------------------
# Rotate 180 degrees about X to make Z axis point downward in world
# desired_quat = R.from_euler('z', -90-24, degrees=True).as_quat()  # [x, y, z, w]

# ------------------------ Inverse Kinematics: position + fixed orientation ------------------------
def compute_inverse_kinematics(model, data, target_pos, target_quat):
    def objective(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ee_site_id]
        ee_quat = data.xquat[ee_site_id]
        pos_error = np.linalg.norm(ee_pos - target_pos)
        quat_error = 1 - np.dot(ee_quat, target_quat)**2
        return pos_error + 0.5 * quat_error  # weighted sum

    bounds = [(model.jnt_range[i, 0], model.jnt_range[i, 1]) for i in range(model.nq)]
    result = minimize(objective, data.qpos.copy(), bounds=bounds, method='SLSQP')
    return result.x if result.success else None

# ------------------------ Logging ------------------------
timevals, q_err_log, ee_force_log = [], [], []
q_desired_log = []

# ------------------------ Control Loop ------------------------
# for i, target_pos in tqdm(enumerate(traj), total=total_frames):
#     desired_quat = R.from_euler('z', traj[i, 3], degrees=True).as_quat()
#     q_desired = compute_inverse_kinematics(model, data, target_pos, desired_quat)

for i in tqdm(range(total_frames)):
    print("yaw:", yaw[i])
    desired_quat = R.from_euler('z', yaw[i], degrees=True).as_quat()
    print(f"Step {i+1}/{total_frames}, Target Position: {traj}")
    q_desired = compute_inverse_kinematics(model, data, traj[i], desired_quat)
    if q_desired is None:
        print(f"[Warning] IK failed at step {i}")
        continue

    q_error = q_desired - data.qpos
    qvel_error = -data.qvel
    data.ctrl[:] = Kp @ q_error + Kd @ qvel_error

    mujoco.mj_step(model, data)
    viewer.sync()

    timevals.append(data.time)
    q_err_log.append(q_error.copy())
    ee_force_log.append(data.cfrc_ext[ee_site_id][:3].copy())
    q_desired_log.append(q_desired.copy())

viewer.close()

# ------------------------ Plot Joint Error ------------------------
timevals = np.array(timevals)
q_err_log = np.array(q_err_log)

# plt.figure(figsize=(12, 6))
# for i in range(model.nq):
#     plt.plot(timevals, q_err_log[:, i], label=f'q{i} error')
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Position Error (rad)")
# plt.title("Joint Position Tracking Error")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ------------------------ Plot Contact Force ------------------------
ee_force_log = np.array(ee_force_log)
plt.figure(figsize=(12, 6))
plt.plot(timevals, ee_force_log[:, 0], label='Fx')
plt.plot(timevals, ee_force_log[:, 1], label='Fy')
plt.plot(timevals, ee_force_log[:, 2], label='Fz')
plt.xlabel("Time (s)")
plt.ylabel("End-Effector Contact Force (N)")
plt.title("End-Effector Contact Force vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------ Plot Desired Joint Positions ------------------------
q_desired_log = np.array(q_desired_log) 
plt.figure(figsize=(12, 6))
for i in range(model.nq):
    plt.plot(timevals, q_desired_log[:, i], label=f'q{i} desired')
plt.xlabel("Time (s)")
plt.ylabel("Desired Joint Position (rad)")
plt.title("Desired Joint Positions Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()