import gym
import numpy as np
import time
import pybullet as p
from hexapod.resources.hexapod import Hexapod
from hexapod.resources.plane import Plane

ORIGINAL_VALUES = []


class HexapodEnv(gym.Env):
    def __init__(self, render=False):
        self.joint_number = 18
        self.buffer_size = 3
        self.servo_high_limit = 2.62
        self.servo_low_limit = -2.62
        self.dt = 1/20
        self.action_space = gym.spaces.box.Box(
            low=np.array([self.servo_low_limit] * self.joint_number, dtype=np.float32),
            high=np.array([self.servo_high_limit] * self.joint_number, dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([self.servo_low_limit] * self.joint_number * 2 * self.buffer_size, dtype=np.float32),
            # 3*18+3*18 = 108
            high=np.array([self.servo_high_limit] * self.joint_number * 2 * self.buffer_size, dtype=np.float32)
        )

        self.np_random, _ = gym.utils.seeding.np_random()
        self.client = p.connect(p.GUI if render else p.DIRECT)

        p.setTimeStep(self.dt, self.client)  # probably, dt is 1/60 sec?

        self._jnt_buffer = np.zeros((self.buffer_size, self.joint_number), dtype=np.float32)
        self._act_buffer = np.zeros((self.buffer_size, self.joint_number), dtype=np.float32)
        self.hexapod = None
        self.done = False
        self.reset()
        self.id = 1
        self.joint_damping_alpha = 0
        self.force_alpha = 0

        # get initial values for Domain Randomization

        for i in range(-1, 18, 1):
            ORIGINAL_VALUES.append(p.getDynamicsInfo(self.id, i))
        # print("original values")
        # print(ORIGINAL_VALUES)

        # get alpha values for simulation tuning

        config_obj = configparser.ConfigParser()

        config_obj.read("configfile.ini") # add ../ on lower dir
        dbparam = config_obj["alpha"]
        self.joint_damping_alpha = float(dbparam['joint_damping'])
        self.force_alpha =  float(dbparam['force'])
        print("self values : ")
        print(self.joint_damping_alpha,self.force_alpha)

    @property
    def get_observation(self):
        # observation is flatten buffer of joint history + action history
        observation = np.concatenate([
            self._jnt_buffer.ravel(),
            self._act_buffer.ravel()
        ])

        return observation

    def non_action_step(self): # just advance one step without any instruction


        p.stepSimulation()
        torques = self.hexapod.get_joint_torques()
        curr_pos, curr_ang = self.hexapod.get_center_position()

        info = {
            # 'reward': reward,
            # 'forward delta': pos_del[1],
            # 'side delta': curr_pos[0],
            'torques': torques
        }
        if np.abs(curr_pos[0]) > 0.5 or curr_pos[2] < 0.05 or np.abs(curr_ang[2]) > 0.5:
            self.done = True

        return self.hexapod.get_joint_values(), self.done, info



    def step(self, action):
        prev_pos, prev_ang = self.hexapod.get_center_position()  # get previous center cartesian and euler for reward

        #print("given")
        #print(action)
        self.hexapod.apply_action(action)  # apply action position on servos
        p.stepSimulation()  # elapse one timestep (above, we assign it as 1/60 s) on pybullet simulation

        #print("got")
        #print(self.hexapod.get_joint_values())


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
        reward = (pos_del[1] + 0.001) / (torque_rms + 1.0) / (np.abs(curr_pos[0]) + 0.5)
        #reward = (pos_del[1] + 0.001) / (np.abs(curr_pos[0]) + 0.5)
        # if current state is unhealthy, then terminate simulation
        # unhealthy if (1) y error is too large (2) or z position is too low (3) or yaw is too large
        if np.abs(curr_pos[0]) > 0.5 or curr_pos[2] < 0.05 or np.abs(curr_ang[2]) > 0.5:
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

    def reset(self, load=False, fixed=False):
        # p.resetSimulation(self.client)
        # p.setGravity(0, 0, -9.8)
        # # Reload the plane and hexapod
        # Plane(self.client)
        # print("start loading hexapod... "); reset_start_time = time.time()  # debug
        # self.hexapod = Hexapod(self.client)
        # reset_end_time = time.time(); print("...end loading hexapod.") # debug
        # print("elapsed time :", reset_end_time - reset_start_time, "sec.\n")  # debug

        if self.hexapod is None:
            p.resetSimulation(self.client)
            p.setGravity(0, 0, -9.8)
            Plane(self.client)
            self.hexapod = Hexapod(self.client)
        else:

            # Domain Randomization PART!!!!!!!!!!!!!!!!
            '''
            print("see Dynamics")
            for i in range(-1,18,1):
            	print(p.getDynamicsInfo(1,i)[0])
            '''
            # 1. Mass Randomization
            # print("mass rand")

            for i in range(-1, 18, 1):
                # print(ORIGINAL_VALUES[i+1][0])

                if load:  # if it is model loading phase, return to original parameters
                    p.changeDynamics(self.id, i, mass=ORIGINAL_VALUES[i + 1][0] * (1),
                                     lateralFriction=ORIGINAL_VALUES[i + 1][1] * (1),
                                     restitution=ORIGINAL_VALUES[i + 1][5], localInertiaDiagonal=(
                        ORIGINAL_VALUES[i + 1][2][0] * (1), ORIGINAL_VALUES[i + 1][2][1] * (1),
                        ORIGINAL_VALUES[i + 1][2][2] * (1)),jointDamping=0.2-self.joint_damping_alpha)
                    self.hexapod.set_joint_forces(np.array([1.5 - self.force_alpha] * 18))
                else:

                    p.changeDynamics(self.id, i, mass=ORIGINAL_VALUES[i + 1][0] * (0.8 + 0.4 * np.random.random()),
                                     lateralFriction=(0.5 + 0.75 * (np.random.random())),
                                     restitution=(0.0001 + 0.8999 * np.random.random()), localInertiaDiagonal=(
                        ORIGINAL_VALUES[i + 1][2][0] * (0.8 + 0.4 * np.random.random()),
                        ORIGINAL_VALUES[i + 1][2][1] * (0.8 + 0.4 * np.random.random()),
                        ORIGINAL_VALUES[i + 1][2][2] * (0.8 + 0.4 * np.random.random())),
                                     jointDamping=0.2-self.joint_damping_alpha, )

                    rand_force_array = np.array([1.5 - self.force_alpha] * 18)
                    for j in range(18):  # init respectively
                        rand_force_array[j] = rand_force_array[j] * (0.8 + 0.4 * np.random.random())

                    self.hexapod.set_joint_forces(rand_force_array)

            self.hexapod.reset_hexapod(fixed=fixed)

        self.done = False
        # reset history buffers
        self._jnt_buffer = np.zeros((self.buffer_size, self.joint_number), dtype=np.float32)
        self._act_buffer = np.zeros((self.buffer_size, self.joint_number), dtype=np.float32)

        return np.array(self.get_observation, dtype=np.float32)

    def render(self, mode='rgbarray', render_size=1000):
        hex_id, client_id = self.hexapod.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=80,
            aspect=1,
            nearVal=0.01,
            farVal=100
        )
        pos, ori = [list(l) for l in p.getBasePositionAndOrientation(hex_id, client_id)]
        pos = np.add(pos, [0.5, 0, 0.1])

        # Rotate camera direction
        rot_mat = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        camera_vec = np.matmul(rot_mat, [-1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 1, 0]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        rgb_array = p.getCameraImage(render_size, render_size, view_matrix, proj_matrix)[2]
        rgb_array = np.reshape(rgb_array, (render_size, render_size, 4))

        return rgb_array

    def close(self):
        p.disconnect(self.client)
