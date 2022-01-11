from hexapod.envs.hexapod_env import HexapodEnv
from hexapod.envs.new_hexapod_env import SimpleHexapodEnv
from hexapod.envs.hexapod_env_2 import revisedHexapodEnv


class HexapodRenderEnv(HexapodEnv):
    def __init__(self):
        HexapodEnv.__init__(self, render=True)


class SimpleHexapodRenderEnv(SimpleHexapodEnv):
    def __init__(self):
        SimpleHexapodEnv.__init__(self, render=True)


class revisedHexapodRenderEnv(revisedHexapodEnv):
    def __init__(self):
        revisedHexapodEnv.__init__(self, gui=True)


class HexapodHexyEnv(revisedHexapodEnv):
    def __init__(self):
        revisedHexapodEnv.__init__(self, isHexy=True)
