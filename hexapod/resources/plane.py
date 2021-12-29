import pybullet as p
import os
from pybullet_data import getDataPath


class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'plane.urdf')
        self.plane = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0],
            physicsClientId=client
        )
        p.setAdditionalSearchPath(getDataPath())
        p.loadTexture('tex256.png')


