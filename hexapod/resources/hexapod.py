import pybullet as p
import os
import math
import numpy as np


class Hexapod:
    def __init__(self, client):
        self.client = client

        self.init_pos = [0.0, 0.0, 0.1]
        self.init_ori = p.getQuaternionFromEuler([np.pi/2, 0.0, 0.0])
        f_name = os.path.join(
            os.path.dirname(__file__),
            'ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf'
        )
        self.hexapod = p.loadURDF(
            fileName=f_name,
            basePosition=self.init_pos,
            baseOrientation=self.init_ori,
            physicsClientId=client
        )

        self.legJoints = range(18)
        self.jointForces = np.array([1.0]*18)

    def get_ids(self):
        return self.hexapod, self.client

    def apply_action(self, action):
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

    def reset_hexapod(self):
        p.resetBasePositionAndOrientation(self.hexapod, self.init_pos, self.init_ori, self.client)
        p.resetBaseVelocity(self.hexapod, [0.0]*3, [0.0]*3, self.client)
        for i in range(p.getNumJoints(self.hexapod, self.client)):
            p.resetJointState(self.hexapod, i, 0.0, 0.0, self.client)
