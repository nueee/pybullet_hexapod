import pybullet as p
import os
import math
import numpy as np


class Hexapod:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(
            os.path.dirname(__file__),
            'ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf'
        )
        self.hexapod = p.loadURDF(
            fileName=f_name,
            basePosition=[0.0, 0.0, 0.1],
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0]),
            physicsClientId=client
        )

        self.legJoints = range(18)
        self.jointForces = np.array([1.0]*18)

    def get_ids(self):
        return self.hexapod, self.client

    def apply_action(self, action):
        p.setJointMotorControl2(
            self.hexapod,
            0,
            controlMode=p.POSITION_CONTROL,
            force=0,
            physicsClientId=self.client
        )
        p.setJointMotorControlArray(
            self.hexapod,
            self.legJoints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=action,
            forces=self.jointForces,
            physicsClientId=self.client
        )

    def get_joint_values(self):
        # Get the joint states
        observation = np.array(p.getJointStates(self.hexapod, self.legJoints, self.client))[:, 0]

        return observation

    def get_center_position(self):
        # get the center cartesian and euler angles
        pos, ang = p.getBasePositionAndOrientation(self.hexapod, self.client)
        ang = p.getEulerFromQuaternion(ang)
        pos, ang = np.array(pos), np.array(ang)

        return pos, ang

    def get_joint_torques(self):
        # get the torques applied on joints
        torques = np.array(p.getJointStates(self.hexapod, self.legJoints, self.client))[:, 3]

        return torques

    # def reset_hexapod(self):
    #     for i in range(p.getNumJoints(self.hexapod, self.client)):
    #         p.resetJointState(self.hexapod, i, 0.0, 0.0, self.client)
    #     p.resetBaseVelocity(self.hexapod, [0.0]*3, [0.0]*3, self.client)
    #     p.resetBasePositionAndOrientation(self.hexapod, [0.0, 0.0, 0.1], [0.0]*4, self.client)
