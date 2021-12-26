import gym
import numpy as np
import time
import pybullet as p
from hexapod.resources.hexapod import Hexapod
from hexapod.resources.plane import Plane
import configparser

'''Domain Randomization Documents(Temporary)
From Jie Tan et.al

mass : 80% ~ 120% (O) 
motor friction 0 ~ 0.05 N m
inertia : 50% ~ 150% (O)
motor strength : 80% ~ 120% 
control step : 3ms ~ 20ms 
latency: 0ms ~ 40ms 
battery Voltage : 14 ~ 16.8V 
contact friction : 0.5 ~ 1.25
IMU bias : +-0.05 rad (X)
IMU Noise(std) 0 ~ 0.05 rad (X)

id value is 1 
original value
[(0.360578160137745, 0.5, (0.0024172112955035325, 0.0040186291109462965, 0.0017457091707458916), (-2.30151070338156e-15, 0.0191096478900804, -0.000877331766132126), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0187184822855231, 0.5, (1.4863322084228813e-05, 1.4864525841291983e-05, 9.801507880531296e-06), (-0.0194026504263946, 0.000962648415416018, -0.0260087497830562), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0551626378910084, 0.5, (5.4098283004696076e-05, 2.484896487305879e-05, 5.0066398421210396e-05), (0.00102679192340011, -0.0223107697760035, -0.000736289104139684), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0652326466286266, 0.5, (8.71175606137404e-05, 0.00010118911246832987, 0.0001642983900955528), (0.0189834041321391, -0.0332649280290424, -0.000415814389983254), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0187184853256138, 0.5, (1.486327955253921e-05, 1.486327360865406e-05, 9.800209879943473e-06), (-0.0194026522515557, 0.000962652233885657, -0.0260088005599141), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.055162636292801, 0.5, (5.409828143732585e-05, 2.4848964153118607e-05, 5.006639697065453e-05), (0.00102679444128451, -0.0223107741231447, -0.000736290752046597), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0652326387878093, 0.5, (8.71175501424057e-05, 0.00010118910030562631, 0.00016429837034725612), (0.0189834134494404, -0.033264933187588, -0.000415814504580459), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0187184822855231, 0.5, (1.4863322084228813e-05, 1.4864525841291983e-05, 9.801507880531296e-06), (-0.0194026504263946, 0.000962648415415969, -0.0260087497830563), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0551626378910084, 0.5, (5.4098283004696076e-05, 2.484896487305879e-05, 5.0066398421210396e-05), (0.00102679192339997, -0.0223107697760034, -0.000736289104140059), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0652326466286265, 0.5, (8.711756061374027e-05, 0.00010118911246832972, 0.00016429839009555257), (0.0189834041321391, -0.0332649280290424, -0.000415814389983629), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0187187999416221, 0.5, (1.4863529371133061e-05, 1.4863529289708585e-05, 9.800380462184052e-06), (-0.0194027231911689, 0.000962720771382118, 0.0260087651240974), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0551628381598268, 0.5, (5.4098479409357443e-05, 2.484905508761386e-05, 5.0066580188051815e-05), (0.00102680024360452, -0.0223108610345546, 0.000736290423570971), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0652327270508921, 0.5, (8.382151186667498e-05, 9.534757524439986e-05, 0.00015516077452587318), (0.018373389650663, -0.0328919722792271, 0.000415796603429955), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0187188019506362, 0.5, (1.4863530966376355e-05, 1.486353088495187e-05, 9.800381514019777e-06), (-0.0194027237249261, 0.00096272374256581, 0.0260088003580916), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0551628383199495, 0.5, (5.40984795663906e-05, 2.48490551597439e-05, 5.006658033338147e-05), (0.00102680218372297, -0.0223108633657772, 0.000736291324398924), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0652327208733654, 0.5, (8.382150392879393e-05, 9.534756621500284e-05, 0.0001551607598321777), (0.0183733977645456, -0.0328919764830271, 0.000415796418803333), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.018718799941622, 0.5, (1.4863529371132983e-05, 1.4863529289708507e-05, 9.800380462183999e-06), (-0.0194027231911689, 0.00096272077138216, 0.0260087651240974), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0551628381598268, 0.5, (5.4098479409357443e-05, 2.484905508761386e-05, 5.0066580188051815e-05), (0.00102680024360452, -0.0223108610345546, 0.000736290423570957), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001), (0.0652327270508919, 0.5, (8.382151186667472e-05, 9.534757524439958e-05, 0.00015516077452587272), (0.0183733896506632, -0.032891972279227, 0.000415796603430107), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2, 0.001)]


Note : 
restitution is not easy parameter to rand

'''

ORIGINAL_VALUES = []
class HexapodEnv(gym.Env):
    def __init__(self, render=False):
        self.joint_number = 18
        self.buffer_size = 3
        self.servo_high_limit = 2.62
        self.servo_low_limit = -2.62
        self.dt = 1/60

        self.action_space = gym.spaces.box.Box(
            low=np.array([self.servo_low_limit]*self.joint_number, dtype=np.float32),
            high=np.array([self.servo_high_limit]*self.joint_number, dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([self.servo_low_limit]*self.joint_number*2*self.buffer_size, dtype=np.float32),  # 3*18+3*18 = 108
            high=np.array([self.servo_high_limit]*self.joint_number*2*self.buffer_size, dtype=np.float32)
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
        # get initial values for Domain Randomization 
        for i in range(-1,18,1):
            ORIGINAL_VALUES.append(p.getDynamicsInfo(self.id,i))
        #print("original values")
        #print(ORIGINAL_VALUES)
        
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
        reward = (pos_del[1] + 0.001) / (torque_rms + 1.0) / (np.abs(curr_pos[0]) + 0.5)

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
        #Plane(self.client)
        # print("start loading hexapod... "); reset_start_time = time.time()  # debug
        # self.hexapod = Hexapod(self.client)
        # reset_end_time = time.time(); print("...end loading hexapod.") # debug
        # print("elapsed time :", reset_end_time - reset_start_time, "sec.\n")  # debug

        if self.hexapod is None:
            p.resetSimulation(self.client)
            p.setGravity(0, 0, -9.8)
            
            # g value setting
            '''
            config_obj = configparser.ConfigParser()
            config_obj.read("../../configfile.ini")
            dbparam = config_obj["postgresql"]
            p.setGravity(0,0,float(dbparam["g"]))
            print("set gravity to" + str(dbparam))
            '''
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
            #print("mass rand")

            for i in range(-1,18,1):
                #print(ORIGINAL_VALUES[i+1][0])
            
                if load: # if it is model loading phase, return to original parameters
                    p.changeDynamics(self.id,i,mass=ORIGINAL_VALUES[i+1][0]*(1),lateralFriction=ORIGINAL_VALUES[i+1][1]*(1),restitution=ORIGINAL_VALUES[i+1][-6],localInertiaDiagonal=(ORIGINAL_VALUES[i+1][-8][0]*(1),ORIGINAL_VALUES[i+1][-8][1]*(1),ORIGINAL_VALUES[i+1][-8][2]*(1)))
                else:
                    p.changeDynamics(self.id,i,mass=ORIGINAL_VALUES[i+1][0]*(0.8+0.4*np.random.random()),lateralFriction=(0.5+0.75*(np.random.random())),restitution=(0.0001+0.8999*np.random.random()),localInertiaDiagonal=(ORIGINAL_VALUES[i+1][-8][0]*(0.8+0.4*np.random.random()),ORIGINAL_VALUES[i+1][-8][1]*(0.8+0.4*np.random.random()),ORIGINAL_VALUES[i+1][-8][2]*(0.8+0.4*np.random.random())))


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
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [-1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 1, 0]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        rgb_array = p.getCameraImage(render_size, render_size, view_matrix, proj_matrix)[2]
        rgb_array = np.reshape(rgb_array, (render_size, render_size, 4))

        return rgb_array


    def close(self):
        p.disconnect(self.client)
