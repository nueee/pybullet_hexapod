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

LEN_SEQ = 10
NUM_SAMP = 10
# command_seq = np.array([[0.25, -0.25, 0.25], [-0.25, 0.25, -0.25]]*(LEN_SEQ//4)
#                        + [[1.7, -1.7, 1.7], [-1.7, 1.7, -1.7]]*(LEN_SEQ//4), dtype=np.float32)
command_seq = np.array([[0.25, -0.25, 0.25], [-0.25, 0.25, -0.25]]*(LEN_SEQ//2), dtype=np.float32)
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


max_torque = 1.5
max_velocity = 4.142414


def get_err(Kp, Kd):
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


def plot_wave(Kp, Kd):
    _, simul_seq = get_err(Kp=Kp, Kd=Kd)

    time = list(map(lambda t: t*dt, range(LEN_SEQ*NUM_SAMP)))

    plt.subplot(131)
    plt.plot(time, simul_seq[:, 0], label='simulation')
    plt.plot(time, sample_seq[:, 0], label='actual')
    plt.title("innermost servo")
    plt.xlabel('time(sec)', fontsize='large')
    plt.ylabel('position(rad)', fontsize='large')

    plt.legend(loc='lower left', fontsize='x-large')

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

Kp = []
opt_Kp = 0.01
Kd = []
opt_Kd = 0.08


f = open('KpKd.csv', 'r')
_ = f.readline()
_ = f.readline()
_ = f.readline()

for i in range(N):
    line = f.readline()
    line = list(map(lambda x: float(x), line.split(',')))
    Kp.append(line[0])
    Kd.append(line[1])
    err_grid.append([])
    for j in range(N):
        err_grid[-1].append(line[j+2])

f.close()
Kp = np.array(Kp)
Kd = np.array(Kd)
err_grid = np.array(err_grid)


print(min_err)
print(opt_Kp)
print(opt_Kd)


fig = plt.figure(1)
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(Kp, Kd)
Z = np.log(err_grid)
ax.plot_surface(X, Y, Z.T, cmap='plasma')
ax.set_xlabel('Kp', fontsize='x-large')
ax.set_ylabel('Kd', fontsize='x-large')
ax.set_zlabel('log(err)', fontsize='x-large')
ax.set_zlim(-3.7, -1.0)


plt.figure(2, figsize=(15, 5))
plot_wave(Kp=opt_Kp, Kd=opt_Kd)
plt.show()

