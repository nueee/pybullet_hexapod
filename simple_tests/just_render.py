import pybullet as p
import time
import numpy as np
import sys
sys.path.append('../')

from hexapod import getDataPath


client = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(getDataPath())

floor = p.loadURDF("../hexapod/resources/plane.urdf")
robot = p.loadURDF(
    fileName="../hexapod/resources/ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf",
    basePosition=[0.0, 0.0, 0.15],
    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0]),
    physicsClientId=client
)

dt = 1. / 30.
p.setTimeStep(dt)
legJoints = range(18)

while True:
    p.setJointMotorControlArray(
        robot,
        legJoints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[0] * 18,
        forces=np.array([200] * 18),
        physicsClientId=client
    )
    # observation = np.array(p.getJointStates(robot, legJoints, client))[:, 0]
    # print(observation)
    p.stepSimulation()
    time.sleep(dt)
