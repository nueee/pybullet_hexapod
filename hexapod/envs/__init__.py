from hexapod.envs.hexapod_env import HexapodEnv
from hexapod.envs.new_hexapod_env import SimpleHexapodEnv


class HexapodRenderEnv(HexapodEnv):

    def __init__(self, render=True):
        HexapodEnv.__init__(self, render=True)


class SimpleHexapodRenderEnv(SimpleHexapodEnv):

    def __init__(self, render=True):
        SimpleHexapodEnv.__init__(self, render=True)
