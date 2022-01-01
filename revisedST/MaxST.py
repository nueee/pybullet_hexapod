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
    plt.plot(time, simul_seq[:, 0], time, sample_seq[:, 0])
    plt.title("innermost servo")

    plt.subplot(132)
    plt.plot(time, simul_seq[:, 1], time, sample_seq[:, 1])
    plt.title("middle servo")

    plt.subplot(133)
    plt.plot(time, simul_seq[:, 2], time, sample_seq[:, 2])
    plt.title("outermost servo")


# --------------- do brute force --------------- #

# N = 20
# err_grid = []
# min_err = 100.0
#
# # max_torque = 1.5
# max_velocity = 4.142414
#
# rt = np.linspace(0, 1, N)
# # rv = np.linspace(0, 1, N)
#
# max_torque = 0.5*rt
# opt_max_torque = 0.0
# # max_velocity = 10.0*rv
# # opt_max_velocity = 0.0
#
# for i in range(N):
#     if i % 10 == 9:
#         print(i)
#     err, _ = get_err(max_torque=max_torque[i], max_velocity=max_velocity)
#     # err, _ = get_err(max_torque=max_torque, max_velocity=max_velocity[i])
#     err_grid.append(err)
#     if err < min_err:
#         min_err = err
#         opt_max_torque = max_torque[i]
#         # opt_max_velocity = max_velocity[i]
#
# print(min_err)
# print(opt_max_torque)
# # print(opt_max_velocity)
#
# iteration = range(N)
# plt.figure(1)
# plt.plot(iteration, err_grid)
# plt.yscale('log')
#
# plt.figure(2, figsize=(15, 5))
# plot_wave(max_torque=opt_max_torque, max_velocity=max_velocity)
# # plot_wave(max_torque=max_torque, max_velocity=opt_max_velocity)
# plt.show()


# ---------- 3D plotting -------- #


N = 101
err_grid = []
min_err = 100.0

rt = np.linspace(0, 1, N)
rv = np.linspace(0, 1, N)

max_torque = 0.2*rt
opt_max_torque = 0.0
max_velocity = 10.0*rv
opt_max_velocity = 0.0

for i in range(N):
    err_grid.append([])
    for j in range(N):
        if (i*N+j) % 10 == 9:
            print(i*N+j)
        err, _ = get_err(max_torque=max_torque[i], max_velocity=max_velocity[j])
        err_grid[-1].append(err)
        if err < min_err:
            min_err = err
            opt_max_torque = max_torque[i]
            opt_max_velocity = max_velocity[j]

print(min_err)
print(opt_max_torque)
print(opt_max_velocity)


# plt.figure(1)
# plt.imshow(err_grid, cm.get_cmap('inferno'))

fig = plt.figure(1)
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(range(N), range(N))
Z = np.array(np.log(err_grid))
ax.plot_surface(X, Y, Z.T, cmap='plasma')
ax.set_xlabel('opt_max_torque')
ax.set_ylabel('opt_max_velocity')
ax.set_zlabel('log(err)')


f = open("MtMv.csv", "w")
f.write("opt_max_torque:"+str(opt_max_torque)+'\n')
f.write("opt_max_velocity:"+str(opt_max_velocity)+'\n')
f.write("max_torque,max_velocity,error(col incr : max_torque incr | row incr : max_velocity incr)\n")
for i in range(N):
    f.write(str(max_torque[i])+','+str(max_velocity[i])+',')
    for j in range(N):
        f.write(str(err_grid[i][j]))
        if j == N-1:
            f.write('\n')
        else:
            f.write(',')
f.close()


plt.figure(2, figsize=(15, 5))
plot_wave(max_torque=opt_max_torque, max_velocity=opt_max_velocity)
plt.show()

