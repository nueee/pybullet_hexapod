import gym
import numpy as np
import time
import pybullet as p
from hexapod.resources.hexapod import Hexapod
from hexapod.resources.plane import Plane


class SimpleHexapodEnv(gym.Env):
    def __init__(self, render=False, dt=0.05):
        self.joint_number = 18
        self.buffer_size = 3
        self.servo_high_limit = []
        self.servo_low_limit = -2.62
        self.dt = dt
        self.action_space = gym.spaces.box.Box(
            low=np.array([self.servo_low_limit] * self.joint_number, dtype=np.float32),
            high=np.array([self.servo_high_limit] * self.joint_number, dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([self.servo_low_limit] * self.joint_number * 2 * self.buffer_size, dtype=np.float32),
            high=np.array([self.servo_high_limit] * self.joint_number * 2 * self.buffer_size, dtype=np.float32)
        )

        self.np_random, _ = gym.utils.seeding.np_random()
        self.client = p.connect(p.DIRECT)

        p.setTimeStep(self.dt, self.client)  # probably, dt is 1/60 sec?
        # p.setPhysicsEngineParameter(numSubSteps=1)

        self._jnt_buffer = np.zeros((self.buffer_size, self.joint_number), dtype=np.float32)
        self._act_buffer = np.zeros((self.buffer_size, self.joint_number), dtype=np.float32)
        self.hexapod = None
        self.done = False

    @property
    def get_observation(self):
        # observation is flatten buffer of joint history + action history
        observation = np.concatenate([
            self._jnt_buffer.ravel(),
            self._act_buffer.ravel()
        ])

        return observation

    def step(self, action):
        prev_pos, prev_ang = self.hexapod.get_center_position()  # get previous center cartesian and euler for reward

        self.hexapod.apply_action(action)  # apply action position on servos
        p.stepSimulation()  # elapse one timestep (above, we assign it as 1/60 s) on pybullet simulation

        # update obs buffer and act buffer
        self._jnt_buffer[1:] = self._jnt_buffer[:-1]
        self._jnt_buffer[0] = self.hexapod.get_joint_values()  # get recent joint values
        self._act_buffer[1:] = self._act_buffer[:-1]
        self._act_buffer[0] = action  # get recent action

        curr_pos, curr_ang = self.hexapod.get_center_position()  # get current center cartesian and euler for reward
        # print("current center position : ", curr_pos)  # debug

        # calculate change of values
        pos_del = curr_pos - prev_pos
        # ang_del = curr_ang - prev_ang  # unused

        torques = self.hexapod.get_joint_torques()
        torque_rms = np.sqrt(np.mean(np.square(torques)))  # get torques applied on joints

        # calculate the reward function
        # (velocity to <+x> + epsilon) / (rms of applied torque + epsilon) / (error to <+-y> + epsilon)
        # each of epsilon will be determined by their corresponding parameter's 'general' dimensions
        reward = (pos_del[1] + 1e-3) / (torque_rms + 0.5) / (np.abs(curr_pos[0]) + 0.5)
        # if current state is unhealthy, then terminate simulation
        # unhealthy if (1) y error is too large (2) or z position is too low (3) or yaw is too large
        if np.abs(curr_pos[0]) > 0.5 or curr_pos[2] < 0.05 or np.max(np.abs(curr_ang)) > 0.5:
            self.done = True

        info = {
            # 'reward': reward,
            # 'forward delta': pos_del[1],
            # 'side delta': curr_pos[0],
            'torques': torques
        }

        return self.get_observation, reward, self.done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.hexapod is None:
            p.resetSimulation(self.client)
            p.setGravity(0, 0, -9.8)
            Plane(self.client)
            self.hexapod = Hexapod(self.client)

            for i in range(self.joint_number):
                p.changeDynamics(
                    1, i,
                    jointLimitForce=1.5,
                    maxJointVelocity=5.0
                )

        self.hexapod.reset_hexapod(offse)

        self.done = False
        # reset history buffers
        self._jnt_buffer = np.array([self.hexapod.joint_pos]*3, dtype=np.float32)
        self._act_buffer = np.array([self.hexapod.joint_pos]*3, dtype=np.float32)

        return np.array(self.get_observation, dtype=np.float32)

    def render(self, render_size=1000):
        hex_id, client_id = self.hexapod.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=100,
            aspect=1,
            nearVal=0.01,
            farVal=100
        )
        pos, ori = [list(i) for i in p.getBasePositionAndOrientation(hex_id, client_id)]
        pos = np.add(pos, [0.5, 0, 0.1])

        # Rotate camera direction
        rot_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        camera_vec = np.matmul(rot_mat, [-1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 1, 0]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        rgb_array = p.getCameraImage(render_size, render_size, view_matrix, proj_matrix)[2]
        rgb_array = np.reshape(rgb_array, (render_size, render_size, 4))

        return rgb_array

    def close(self):
        p.disconnect(self.client)
