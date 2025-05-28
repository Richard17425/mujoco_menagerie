'''
task space Impedance control, 但目前没法让机械臂按照设定轨迹移动, 末端关节无法竖直向下
仿真过程的output在img文件夹里, torque代表的是actuator input
'''
import mujoco  
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco_viewer

class ImpedanceController(mujoco_viewer.CustomViewer):
    def __init__(self, model_path):
        super().__init__(model_path, distance=3.0, azimuth=-45, elevation=-30)

    def runBefore(self):
        self.Kp = np.diag([10.0, 10.0, 10.0])
        self.Kd = np.diag([4.0, 4.0, 4.0])
        
        self.Kp_rot = np.diag([100.0, 100.0, 100.0])


        # traj
        self.start_pos = np.array([0.1, 0.3, 0.15])
        self.end_pos = np.array([0.1,  -0.3, 0.15])
        self.total_time = 5.0
        self.dt = self.model.opt.timestep
        self.num_steps = int(self.total_time / self.dt)
        self.current_step = 0

        # record data
        self.q_history = np.zeros((self.num_steps, self.model.nu))
        self.qdot_history = np.zeros((self.num_steps, self.model.nu))
        self.torque_history = np.zeros((self.num_steps, self.model.nu))

        # get end-effector site ID
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # pos setup
        self.initial_qpos = np.array([1.51, -1.57, -1.79, -2.2, 1.51, 0.0])
        self.data.qpos[:] = self.initial_qpos
        mujoco.mj_forward(self.model, self.data)

    def runFunc(self):
        if self.current_step >= self.num_steps:
            return

        t = self.current_step * self.dt
        x_des = self.start_pos + (self.end_pos - self.start_pos) * (t / self.total_time)
        xdot_des = (self.end_pos - self.start_pos) / self.total_time

        # current pos and vel
        ee_pos = self.data.xpos[self.ee_site]
        ee_vel = self.data.cvel[self.ee_site][3:]

        pos_err = x_des - ee_pos
        vel_err = xdot_des - ee_vel

        F = self.Kp @ pos_err + self.Kd @ vel_err
        print(f"Step {self.current_step}, Position Error: {pos_err}, Velocity Error: {vel_err}, Force: {F}")

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        # rotm = self.data.xmat[self.ee_site].reshape(3, 3)
        # R_cur = rotm
        # z_des = np.array([0, 0, -1])     # 工具 z 轴朝地面
        # x_des = np.array([1, 0, 0])      # 工具 x 轴朝前（任意，与 z 正交）
        # y_des = np.cross(z_des, x_des)   # 自动求出 y 轴

        # R_des = np.column_stack((x_des, y_des, z_des))  # 构建目标姿态

        # # 姿态误差（SO(3) 近似）
        # R_err = R_des @ R_cur.T
        # rot_err = 0.5 * (np.cross(R_cur[:, 0], R_des[:, 0]) +
        #                 np.cross(R_cur[:, 1], R_des[:, 1]) +
        #                 np.cross(R_cur[:, 2], R_des[:, 2]))

        # Fr = self.Kp_rot @ rot_err  # 控制姿态的力矩
        # print(f"Fr {Fr}, Rotation Error: {rot_err}")

        # 计算总控制力矩
        # F_total = np.concatenate([F, Fr])  # F 是位置力，Fr 是姿态力矩
        # J_full = np.vstack([jacp, jacr])
        torque = jacp.T @ F
        print("torque:",torque)

        self.data.ctrl[:3] = torque[:3]
        self.data.ctrl[-3:] = self.initial_qpos[-3:]

        self.q_history[self.current_step] = self.data.qpos[:self.model.nu]
        self.qdot_history[self.current_step] = self.data.qvel[:self.model.nu]
        self.torque_history[self.current_step] = self.data.ctrl[:self.model.nu]

        self.current_step += 1

        # plot
        if self.current_step == self.num_steps:
            time = np.arange(0, self.total_time, self.dt)
            plt.figure(figsize=(12, 8))

            # joint angles
            plt.subplot(3, 1, 1)
            for j in range(self.model.nu):
                plt.plot(time, self.q_history[:, j], label=f'Joint {j+1}')
            plt.title('Joint Angles')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (rad)')
            plt.legend()

            # joint velocities
            plt.subplot(3, 1, 2)
            for j in range(self.model.nu):
                plt.plot(time, self.qdot_history[:, j], label=f'Joint {j+1}')
            plt.title('Joint Velocities')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (rad/s)')
            plt.legend()

            # control torques
            plt.subplot(3, 1, 3)
            for j in range(self.model.nu):
                plt.plot(time, self.torque_history[:, j], label=f'Joint {j+1}')
            plt.title('Control Torques')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (N.m)')
            plt.legend()

            plt.tight_layout()
            plt.savefig("img/ur10e_impedance_control_result.png", dpi=600)

if __name__ == "__main__":
    controller = ImpedanceController("universal_robots_ur10e/scene_Door.xml")
    controller.run_loop()