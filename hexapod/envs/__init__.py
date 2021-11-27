from hexapod.envs.hexapod_env import HexapodEnv


class HexapodRenderEnv(HexapodEnv):

    def __init__(self, render=True):
        HexapodEnv.__init__(self, render=True)