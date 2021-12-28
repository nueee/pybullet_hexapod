import sys
import numpy as np
import pybullet as p

sys.path.append('../')
from hexapod import getDataPath


NUM_DXL = 18
dt = 0.002
action_ratio = 50
legJoints = range(NUM_DXL)

LEN_SEQ = 20
NUM_SAMP = 10
command_seq = np.array([[0.5]*NUM_DXL, [-0.5]*NUM_DXL]*(LEN_SEQ//2), dtype=np.float32)
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


f = open('sampled.csv', 'r')

for i in range(LEN_SEQ):
    for j in range(NUM_SAMP):
        line = f.readline()
        line = list(map(lambda x: float(x), line.split(',')))
        sample_seq[i][j] = line

f.close()


error_seq = np.zeros((LEN_SEQ, NUM_SAMP, NUM_DXL), dtype=np.float32)

for i in range(LEN_SEQ):
    p.setJointMotorControlArray(
        robot,
        legJoints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=command_seq[i],
        forces=np.array([1.5]*NUM_DXL),
        physicsClientId=client
    )
    for j in range(NUM_SAMP):
        timer = 0.0
        for k in range(NUM_DXL):
            p.stepSimulation(dt)
            timer += dt
            joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][k]
            error_seq[i][j][k] = sample_seq[i][j][k] - joint_pos
        
