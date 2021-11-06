import gym
import numpy as np
import math
import pybullet as p
from hexapod.resources.hexapod import Hexapod
from hexapod.resources.plane import Plane
import matplotlib.pyplot as plt


class HexapodEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
                         dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                          dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
                         dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                          dtype=np.float32)
        )
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/60, self.client)

        self.hexapod = None
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.prev_pos_x = None
        self.reset()

    def step(self, action):
        self.hexapod.apply_action(action)
        p.stepSimulation()
        hex_obs = self.hexapod.get_observation()

        curr_pos, curr_ang = self.hexapod.get_position()
        reward = self.prev_pos_x - curr_pos[0]
        self.prev_pos_x = curr_pos[0]

        if np.abs(curr_pos[1]) > 0.3 or curr_pos[2] < 0.03:
            self.done = True

        return hex_obs, reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        # Reload the plane and hexapod
        Plane(self.client)
        self.hexapod = Hexapod(self.client)

        self.done = False
        self.prev_pos_x = 0.0

        # Get observation to return
        hex_obs = self.hexapod.get_observation()

        print(hex_obs)

        return np.array(hex_obs, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        hex_id, client_id = self.hexapod.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(hex_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
