import sys
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
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


Kp = 0.01
Kd = 0.08

def get_err(max_torque, max_velocity):
    for dxl in legJoints:
        p.resetJointState(robot, dxl, 0.0, 0.0, client)
        p.changeDynamics(
            robot, dxl,
            jointLimitForce=max_torque,
            maxJointVelocity=max_velocity
        )

    error_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)
    simul_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)

    for l in range(LEN_SEQ):
        for joint in legJoints:
            p.setJointMotorControl2(
                robot, joint,
                controlMode=p.POSITION_CONTROL,
                positionGain=Kp,
                velocityGain=Kd,
                targetPosition=command_seq[l][joint],
                force=max_torque,
                maxVelocity=max_velocity,
                physicsClientId=client
            )
        for m in range(NUM_SAMP):
            for n in legJoints:
                joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][n]
                simul_seq[l*NUM_SAMP+m][n] = joint_pos
                error_seq[l*NUM_SAMP+m][n] = np.abs(sample_seq[l*NUM_SAMP+m][n] - joint_pos)
            p.stepSimulation()

    return np.max(np.mean(error_seq, axis=0)), simul_seq


def plot_wave(max_torque, max_velocity):
    err, simul_seq = get_err(max_torque=max_torque, max_velocity=max_velocity)

    time = list(map(lambda t: t*dt, range(LEN_SEQ*NUM_SAMP)))

    plt.subplot(131)
    plt.plot(time, simul_seq[:, 0], label='simulation')
    plt.plot(time, sample_seq[:, 0], label='actual')
    plt.title("innermost servo")
    plt.xlabel('time(sec)', fontsize='large')
    plt.ylabel('position(rad)', fontsize='large')

    plt.legend(loc='upper left', fontsize='x-large')

    plt.subplot(132)
    plt.plot(time, simul_seq[:, 1], time, sample_seq[:, 1])
    plt.title("middle servo")

    plt.subplot(133)
    plt.plot(time, simul_seq[:, 2], time, sample_seq[:, 2])
    plt.title("outermost servo")


# ---------- 3D plotting -------- #


N = 101
err_grid = []
min_err = 100.0

max_torque = []
opt_max_torque = 0.03
max_velocity = []
opt_max_velocity = 5.3


f = open('MtMv.csv', 'r')
_ = f.readline()
_ = f.readline()
_ = f.readline()

for i in range(N):
    line = f.readline()
    line = list(map(lambda x: float(x), line.split(',')))
    max_torque.append(line[0])
    max_velocity.append(line[1])
    err_grid.append([])
    for j in range(N):
        err_grid[-1].append(line[j+2])

f.close()
max_torque = np.array(max_torque)
max_velocity = np.array(max_velocity)
err_grid = np.array(err_grid)


print(min_err)
print(opt_max_torque)
print(opt_max_velocity)


fig = plt.figure(1)
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(max_torque, max_velocity)
Z = np.log(err_grid)
ax.plot_surface(X, Y, Z.T, cmap='plasma')
ax.set_xlabel('Kp', fontsize='x-large')
ax.set_ylabel('Kd', fontsize='x-large')
ax.set_zlabel('log(err)', fontsize='x-large')


plt.figure(2, figsize=(15, 5))
plot_wave(max_torque=opt_max_torque, max_velocity=opt_max_velocity)
plt.show()

