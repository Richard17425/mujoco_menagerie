'''
Impedance control 维持静态位姿的一个测试代码。可以在窗口里双击选择机械臂关节 然后ctrl+左键拖动机械臂移动
控制器会保证机械臂偏离desire pose之后回到原位
仿真过程的output在img文件夹里, torque代表的是actuator input
'''
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import time
import mujoco_viewer
import numpy as np

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):

        self.Kp = np.diag([100] * self.model.nu)
        self.Kd = np.diag([10] * self.model.nu)

        # self.q_desired = np.zeros(self.model.nu)
        self.q_desired = [0.0, -0.991, 0.196, 0.662, -0.88, 0.66]

        # parameters for simulation
        self.total_time = 10  # total simulation time [seconds]
        self.dt = self.model.opt.timestep  # time step of the simulation [seconds]
        self.num_steps = int(self.total_time / self.dt)

        # recording data
        self.q_history = np.zeros((self.num_steps, self.model.nu))
        self.qdot_history = np.zeros((self.num_steps, self.model.nu))
        self.torque_history = np.zeros((self.num_steps, self.model.nu))
        self.index = 0
    
    def runFunc(self):
        if self.index >= self.num_steps:
            return  # exit if we have reached the number of steps
        # reading current joint positions and velocities
        q = self.data.qpos[:self.model.nu]
        qdot = self.data.qvel[:self.model.nu]

        # calculating control torque using impedance control
        error = self.q_desired - q
        print(self.index, self.num_steps, self.model.nu, error)
        torque = self.Kp @ error - self.Kd @ qdot

        self.data.ctrl[:] = torque

        self.q_history[self.index] = q
        self.qdot_history[self.index] = qdot
        self.torque_history[self.index] = torque
        self.index += 1

        if self.index == self.num_steps:

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
            plt.savefig("img/impedance_control_result.png", dpi=600)
            # plt.show()

# test = Test("model/trs_so_arm100/scene_without_position.xml")
test = Test("universal_robots_ur10e/scene_Door.xml")
test.run_loop()

