'''
task space Impedance control, 
这个单纯为了测试新的ik方法和解决末端关节不能竖直向下的问题
但是机械臂在发癫
'''

import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco_viewer
from scipy.optimize import minimize

class ImpedanceController(mujoco_viewer.CustomViewer):
    def __init__(self, model_path):
        # Initialize the viewer with a fixed camera pose
        super().__init__(model_path, distance=3.0, azimuth=-45, elevation=-30)

        # Joint-space PD gains: higher for position joints, lower for orientation joints
        self.Kp_joint = np.diag([40.0, 40.0, 40.0, 20.0, 20.0, 20.0])
        self.Kd_joint = np.diag([10.0, 10.0, 10.0,  5.0,  5.0,  5.0])

        # End-effector linear trajectory start and end in Cartesian space
        self.start_pos  = np.array([-0.1,  0.4, 0.15])
        self.end_pos    = np.array([-0.1, -0.4, 0.15])
        self.total_time = 5.0  # seconds

        # Simulation time stepping
        self.dt          = self.model.opt.timestep
        self.num_steps   = int(self.total_time / self.dt)
        self.current_step = 0

        # Data buffers for plotting after sim ends
        self.q_history      = np.zeros((self.num_steps, self.model.nu))
        self.qdot_history   = np.zeros((self.num_steps, self.model.nu))
        self.torque_history = np.zeros((self.num_steps, self.model.nu))

        # Locate the end-effector site by name
        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

        # Initial joint configuration guess
        self.initial_qpos = np.array([1.51, -1.57, -1.79, -2.20, 1.51, 0.0])
        # Align initial_qpos so that the EE starts exactly at start_pos
        self.data.qpos[:] = self.initial_qpos
        mujoco.mj_forward(self.model, self.data)
        # Compute IK to snap to exact start_pos
        q0 = self.initial_qpos.copy()
        self.initial_qpos = self.compute_inverse_kinematic(self.start_pos, q0)
        self.data.qpos[:] = self.initial_qpos
        mujoco.mj_forward(self.model, self.data)

    def compute_inverse_kinematic(self, target_pos, q_init):
        """
        Solve IK by minimizing ||fk(q) - target_pos|| using BFGS.
        Returns the joint vector q that places the EE at target_pos.
        """
        # Backup current qpos
        q_backup = self.data.qpos[:self.model.nu].copy()

        def fk_error(q):
            # Temporarily set q, compute forward kinematics, measure distance
            self.data.qpos[:self.model.nu] = q
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.xpos[self.ee_site]
            return np.linalg.norm(ee_pos - target_pos)

        res = minimize(
            fk_error, q_init,
            method='BFGS',
            options={'gtol': 1e-6, 'maxiter': 200}
        )

        # Restore qpos to backup
        self.data.qpos[:self.model.nu] = q_backup
        mujoco.mj_forward(self.model, self.data)
        return res.x

    def runFunc(self):
        if self.current_step >= self.num_steps:
            return

        # Compute desired Cartesian position via linear interpolation
        t     = self.current_step * self.dt
        x_des = self.start_pos + (self.end_pos - self.start_pos) * (t / self.total_time)

        # Inverse kinematics: get desired joint configuration q_des
        q_init = self.data.qpos[:self.model.nu].copy()
        q_des  = self.compute_inverse_kinematic(x_des, q_init)

        # Read current joint positions & velocities
        q_cur    = self.data.qpos[:self.model.nu]
        qdot_cur = self.data.qvel[:self.model.nu]

        # Joint-space PD torque command
        tau = self.Kp_joint @ (q_des - q_cur) - self.Kd_joint @ qdot_cur

        # Gravity compensation via inverse dynamics
        mujoco.mj_inverse(self.model, self.data)  
        gravity_torque = self.data.qfrc_inverse[:self.model.nu]
        tau += gravity_torque

        # Apply the total torque command
        self.data.ctrl[:] = tau

        # Record for later plotting
        self.q_history[self.current_step]      = q_cur
        self.qdot_history[self.current_step]   = qdot_cur
        self.torque_history[self.current_step] = tau

        self.current_step += 1

        # After final step, plot and save results
        if self.current_step == self.num_steps:
            time = np.linspace(0, self.total_time, self.num_steps)

            plt.figure(figsize=(10, 8))

            # Joint angles over time
            plt.subplot(3, 1, 1)
            for j in range(self.model.nu):
                plt.plot(time, self.q_history[:, j], label=f'Joint {j+1}')
            plt.title('Joint Angles')
            plt.ylabel('Angle (rad)')
            plt.legend()

            # Joint velocities over time
            plt.subplot(3, 1, 2)
            for j in range(self.model.nu):
                plt.plot(time, self.qdot_history[:, j], label=f'Joint {j+1}')
            plt.title('Joint Velocities')
            plt.ylabel('Velocity (rad/s)')
            plt.legend()

            # Control torques over time
            plt.subplot(3, 1, 3)
            for j in range(self.model.nu):
                plt.plot(time, self.torque_history[:, j], label=f'Joint {j+1}')
            plt.title('Control Torques')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (N·m)')
            plt.legend()

            plt.tight_layout()
            plt.savefig("img/ur10e_impedance_control_result.png", dpi=600)
            print("Simulation complete. Results saved to img/ur10e_impedance_control_result.png")

if __name__ == "__main__":
    controller = ImpedanceController("universal_robots_ur10e/scene_Door.xml")
    controller.run_loop()
