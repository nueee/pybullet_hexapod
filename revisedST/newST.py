import sys
import numpy as np
import pybullet as p

sys.path.append('../')
from hexapod import getDataPath


NUM_DXL = 18
dt = 0.0025
action_ratio = 50
legJoints = range(NUM_DXL)

LEN_SEQ = 20
NUM_SAMP = 10
command_seq = np.array([[0.27, -0.27, 0.27], [-0.27, 0.27, -0.27]]*(LEN_SEQ//4)
                       + [[1.5, -1.5, 1.5], [-1.5, 1.5, -1.5]]*(LEN_SEQ//4), dtype=np.float32)
sample_seq = np.zeros((LEN_SEQ, NUM_SAMP, NUM_DXL), dtype=np.float32)


client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.8)

p.setAdditionalSearchPath(getDataPath())
floor = p.loadURDF('../hexapod/resources/plane.urdf')
robot = p.loadURDF(
    fileName='../hexapod/resources/ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf',
    basePosition=[0.0, 0.0, 0.5],
    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0]),
    physicsClientId=client
)

p.setTimeStep(dt)


f = open('new_sampled.csv', 'r')

for i in range(LEN_SEQ):
    for j in range(NUM_SAMP):
        line = f.readline()
        line = list(map(lambda x: float(x), line.split(',')))
        sample_seq[i][j] = line

f.close()


def get_err():
    error_seq = np.zeros((LEN_SEQ, NUM_SAMP, NUM_DXL), dtype=np.float32)

    for l in range(LEN_SEQ):
        p.setJointMotorControlArray(
            robot,
            legJoints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=command_seq[i],
            forces=np.array([1.5]*NUM_DXL),
            physicsClientId=client
        )
        for m in range(NUM_SAMP):
            timer = 0.0
            for n in range(NUM_DXL):
                p.stepSimulation(dt)
                timer += dt
                joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][k]
                error_seq[l][m][n] = sample_seq[l][m][n] - joint_pos
            for n in range(9):
                p.stepSimulation(dt)

    return error_seq


N = 100
T = 100
scale = np.sqrt(T)
start = np.random()
