import sys
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import time

sys.path.append('../')
from hexapod import getDataPath


NUM_DXL = 3
dt = 0.025
legJoints = range(NUM_DXL)

LEN_SEQ = 20
NUM_SAMP = 10
command_seq = np.array([[0.27, -0.27, 0.27], [-0.27, 0.27, -0.27]]*(LEN_SEQ//4)
                       + [[1.5, -1.5, 1.5], [-1.5, 1.5, -1.5]]*(LEN_SEQ//4), dtype=np.float32)
# command_seq = np.array([[0.27, -0.27, 0.27], [-0.27, 0.27, -0.27]]*(LEN_SEQ//2), dtype=np.float32)
sample_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)


client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.8)
# p.setGravity(0, 0, 0)

p.setAdditionalSearchPath(getDataPath())
floor = p.loadURDF('../hexapod/resources/plane.urdf')
robot = p.loadURDF(
    fileName='../hexapod/resources/ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf',
    basePosition=[0.0, 0.0, 0.1],
    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0]),
    physicsClientId=client
)

p.setTimeStep(dt)
p.setPhysicsEngineParameter(numSubSteps=10)

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


f = open('new_sampled.csv', 'r')

for i in range(LEN_SEQ):
    for j in range(NUM_SAMP):
        line = f.readline()
        line = list(map(lambda x: float(x), line.split(',')))
        sample_seq[i*NUM_SAMP+j] = line

f.close()


def plot_wave(max_torque, max_vel):
    for rj in range(p.getNumJoints(robot, client)):
        p.resetJointState(robot, rj, 0.0, 0.0, client)

    for dxl in range(NUM_DXL):
        p.changeDynamics(
            robot, dxl,
            jointLimitForce=max_torque[dxl],
            maxJointVelocity=max_vel
        )

    simul_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)

    for l in range(LEN_SEQ):
        p.setJointMotorControlArray(
            robot,
            legJoints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=command_seq[l],
            forces=np.array(max_torque),
            physicsClientId=client
        )
        for m in range(NUM_SAMP):
            for n in range(NUM_DXL):
                joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][n]
                simul_seq[l*NUM_SAMP+m][n] = joint_pos
            p.stepSimulation()

    time = list(map(lambda t: t*0.025, range(LEN_SEQ*NUM_SAMP)))

    plt.subplot(131)
    plt.plot(time, simul_seq[:, 0], time, sample_seq[:, 0])
    plt.title("innermost servo")

    plt.subplot(132)
    plt.plot(time, simul_seq[:, 1], time, sample_seq[:, 1])
    plt.title("middle servo")

    plt.subplot(133)
    plt.plot(time, simul_seq[:, 2], time, sample_seq[:, 2])
    plt.title("outermost servo")


def get_err(max_torque):
    for rj in range(p.getNumJoints(robot, client)):
        p.resetJointState(robot, rj, 0.0, 0.0, client)

    error_seq = np.zeros((LEN_SEQ*NUM_SAMP, NUM_DXL), dtype=np.float32)

    for l in range(LEN_SEQ):
        p.setJointMotorControlArray(
            robot,
            legJoints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=command_seq[l],
            forces=max_torque,
            physicsClientId=client
        )
        for m in range(NUM_SAMP):
            for n in range(NUM_DXL):
                joint_pos = np.array(p.getJointStates(robot, legJoints, client))[:, 0][n]
                error_seq[l*NUM_SAMP+m][n] = np.abs(sample_seq[l*NUM_SAMP+m][n] - joint_pos)
            p.stepSimulation()
            # time.sleep(dt)

    return np.max(np.mean(error_seq, axis=0))


N = 20
err_grid = [[], [], []]
min_err = 100.0

# mv = [[], [], []]
# max_torque = [[], [], []]
# for d in range(NUM_DXL):
#     mv[d] = np.linspace(0, 1, N)
#     max_torque[d] = (mv[d]-0.5)*0.06
# opt_max_tor = [1.5, 1.5, 1.5]
#
# for d in [2, 1, 0]:
#     for i in range(N):
#         p.changeDynamics(
#             robot, d,
#             jointLimitForce=max_torque[d][i],
#             maxJointVelocity=4.142414
#         )
#         print(i)
#         if d == 2:
#             err = np.max(get_err(max_torque=[opt_max_tor[0], opt_max_tor[1], max_torque[2][i]]))
#         elif d == 1:
#             err = np.max(get_err(max_torque=[opt_max_tor[0], max_torque[1][i], opt_max_tor[2]]))
#         else:
#             err = np.max(get_err(max_torque=[max_torque[0][i], opt_max_tor[1], opt_max_tor[2]]))
#         err_grid[d].append(err)
#         if err < min_err:
#             min_err = err
#             opt_max_tor[d] = max_torque[d][i]
#     print(min_err)
#     print(opt_max_tor)
#     p.changeDynamics(
#         robot, d,
#         jointLimitForce=opt_max_tor[d],
#         maxJointVelocity=4.142414
#     )

# iteration = range(N)
# plt.figure(1)
# plt.plot(iteration, err_grid[0], iteration, err_grid[1], iteration, err_grid[2])

plt.figure(2)
plot_wave(max_torque=[0.04, 0.03, 1.5], max_vel=5)
plt.show()


# N = 100
# err_grid = []
# min_err = 100.0

# jv = np.linspace(0, 1, N)
# jnt_dmp = np.float_power(10, -1 * jv - 0.5)
# opt_jnt_dmp = 0.0
#
# for i in range(N):
#     for j in range(NUM_DXL):
#         p.changeDynamics(
#             robot, j,
#             jointLimitForce=1.5,
#             maxJointVelocity=4.142414,
#             jointDamping=jnt_dmp[i]
#         )
#     print(i)
#     err = np.max(get_err(max_torque=1.5))
#     err_grid.append(err)
#     if err < min_err:
#         min_err = err
#         opt_jnt_dmp = jnt_dmp[i]
#
# print(min_err)
# print(opt_jnt_dmp)
#
# iteration = range(N)
# plt.figure(1)
# plt.plot(iteration, err_grid)

# plt.figure(2)
# plot_wave_for_dmp(joint_damping=0.1, max_vel=6)
# plt.show()


# mv = np.linspace(0, 1, N)
# max_tor = 10*mv+1e-2
# opt_max_tor = 0.0
#
# for i in range(N):
#     for j in range(NUM_DXL):
#         p.changeDynamics(
#             robot, j,
#             jointLimitForce=max_tor[i]
#         )
#     print(i)
#     err = get_err(max_torque=max_tor[i])
#     err_grid.append(err)
#     if err < min_err:
#         min_err = err
#         opt_max_tor = max_tor[i]
#
# print(min_err)
# print(opt_max_tor)
#
# iteration = range(N)
# plt.plot(iteration, err_grid)
# plt.show()


# vv = np.linspace(0, 1, N)
# max_vel = 10*vv+1e-3
# opt_max_vel = 0.0
#
# for i in range(N):
#     for j in range(NUM_DXL):
#         p.changeDynamics(
#             robot, j,
#             jointLimitForce=1.5,
#             maxJointVelocity=max_vel[i]
#         )
#     print(i)
#     err = np.max(get_err(max_torque=1.5))
#     err_grid.append(err)
#     if err < min_err:
#         min_err = err
#         opt_max_vel = max_vel[i]
#
# print(min_err)
# print(opt_max_vel)
#
# iteration = range(N)
#
# plt.figure(1)
# plt.plot(iteration, err_grid)
#
# plt.figure(2)
# plot_wave_for_vel(opt_max_vel)
# plt.show()

# N = 1000
# T = 1000
# alpha = 0.99
# scale = np.sqrt(T)
#
# print("default error : ", get_err())
#
# jv = np.random.rand()
# # mv = np.random.rand()
#
# # mv+1.0
#
# jnt_dmp = np.float_power(10, -5 * jv + 0.5)
# for jnt in range(NUM_DXL):
#     p.changeDynamics(
#         robot,
#         jnt,
#         jointDamping=jnt_dmp
#     )
# old_err = get_err()
# err_history.append(old_err)
#
# print("initial error : ", old_err)
#
# for it in range(N):
#     new_jv = jv + np.random.rand()
#     # new_mv = mv + np.random.rand()
#
#     if new_jv > 1 or new_jv < 0:
#         new_jv = jv
#     else:
#         jnt_dmp = np.float_power(10, -5 * jv + 0.5)
#         for jnt in range(NUM_DXL):
#             p.changeDynamics(
#                 robot,
#                 jnt,
#                 jointDamping=jnt_dmp
#             )
#         new_err = get_err()
#         if np.log(np.random.rand())*T > (old_err - new_err):
#             new_jv = jv
#         else:
#             old_err = new_err
#             print("error at ", it+1, " : ", old_err)
#
#     jv = new_jv
#     T = alpha*T
#     err_history.append(old_err)
#
# iteration = range(N+1)
# plt.plot(iteration, err_history)
# plt.show()
