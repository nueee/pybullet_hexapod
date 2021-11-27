from gym.envs.registration import register
import os

register(
    id='Hexapod-v0',
    entry_point='hexapod.envs:HexapodEnv'
)

register(
    id='HexapodRender-v0',
    entry_point='hexapod.envs:HexapodRenderEnv'
)


def getDataPath():

    return os.path.join(os.path.dirname(__file__))
