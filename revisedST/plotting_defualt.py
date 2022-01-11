import sys
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import time

sys.path.append('../')
from hexapod import getDataPath


# --------------- ready variables --------------- #


NUM_DXL = 3
dt = 0.025
legJoints = range(NUM_DXL)

LEN_SEQ = 20
NUM_SAMP = 10
command_seq = np.array([[0.25, -0.25, 0.25], [-0.25, 0.25, -0.25]]*(LEN_SEQ//4)
                       + [[1.7, -1.7, 1.7], [-1.7, 1.7, -1.7]]*(LEN_SEQ//4), dtype=np.float32)
# command_seq = np.array([[0.25, -0.25, 0.25], [-0.25, 0.25, -0.25]]*(LEN_SEQ//2), dtype=np.float32)
sample_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)


# --------------- initialize pybullet --------------- #


client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(getDataPath())
floor = p.loadURDF('../hexapod/resources/plane.urdf', useFixedBase=True)
robot = p.loadURDF(
    fileName='../hexapod/resources/ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf',
    basePosition=[0.0, 0.0, 0.1],
    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0]),
    physicsClientId=client
)

p.setTimeStep(dt)
p.setPhysicsEngineParameter(numSubSteps=6)

p.createConstraint(
    parentBodyUniqueId=p.getBodyUniqueId(robot),
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0.5],
    childFrameOrientation=p.getQuaternionFromEuler([np.pi / 2, 0.0, 0.0]),
)


f = open('new_sampled2.csv', 'r')

for i in range(LEN_SEQ):
    for j in range(NUM_SAMP):
        line = f.readline()
        line = list(map(lambda x: float(x), line.split(',')))
        sample_seq[i*NUM_SAMP+j] = line

f.close()


# --------------- define function --------------- #


def get_err_d():
    for dxl in legJoints:
        p.resetJointState(robot, dxl, 0.0, 0.0, client)

    error_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)
    simul_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)

    for l in range(LEN_SEQ):
        for joint in legJoints:
            p.setJointMotorControl2(
                robot, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=command_seq[l][joint],
                physicsClientId=client
            )
        for m in range(NUM_SAMP):
            for n in legJoints:
                joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][n]
                simul_seq[l*NUM_SAMP+m][n] = joint_pos
                error_seq[l*NUM_SAMP+m][n] = np.abs(sample_seq[l*NUM_SAMP+m][n] - joint_pos)
            p.stepSimulation()

    return np.max(np.mean(error_seq, axis=0)), simul_seq


def get_err_s():
    for dxl in legJoints:
        p.resetJointState(robot, dxl, 0.0, 0.0, client)
        p.changeDynamics(
            robot, dxl,
            jointLimitForce=0.03,
            maxJointVelocity=5.3
        )

    error_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)
    simul_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)

    for l in range(LEN_SEQ):
        for joint in legJoints:
            p.setJointMotorControl2(
                robot, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=command_seq[l][joint],
                positionGain=0.01,
                velocityGain=0.08,
                force=0.03,
                maxVelocity=5.3,
                physicsClientId=client
            )
        for m in range(NUM_SAMP):
            for n in legJoints:
                joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][n]
                simul_seq[l*NUM_SAMP+m][n] = joint_pos
                error_seq[l*NUM_SAMP+m][n] = np.abs(sample_seq[l*NUM_SAMP+m][n] - joint_pos)
            p.stepSimulation()

    return np.max(np.mean(error_seq, axis=0)), simul_seq


# ---------- plotting -------- #


plt.figure(2, figsize=(15, 5))

_, simul_seq_default = get_err_d()
_, simul_seq_ST = get_err_s()

time = list(map(lambda t: t*dt, range(LEN_SEQ*NUM_SAMP)))

plt.subplot(131)
plt.plot(time, simul_seq_ST[:, 0], 'r', label='w/ ST sim')
plt.plot(time, simul_seq_default[:, 0], 'b', label='w/o ST sim')
plt.plot(time, sample_seq[:, 0], 'k--', label='actual')
plt.title("innermost servo")
plt.ylabel('position(rad)', fontsize='large')

plt.legend(loc='lower left', fontsize='large')

plt.subplot(132)
plt.plot(time, simul_seq_ST[:, 1], 'r')
plt.plot(time, simul_seq_default[:, 1], 'b')
plt.plot(time, sample_seq[:, 1], 'k--')
plt.title("middle servo")
plt.xlabel('time(sec)', fontsize='large')

plt.subplot(133)
plt.plot(time, simul_seq_ST[:, 2], 'r')
plt.plot(time, simul_seq_default[:, 2], 'b')
plt.plot(time, sample_seq[:, 2], 'k--')
plt.title("outermost servo")

plt.tight_layout()
plt.show()

