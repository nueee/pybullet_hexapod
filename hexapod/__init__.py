from gym.envs.registration import register
register(
    id='Hexapod-v0',
    entry_point='hexapod.envs:HexapodEnv'
)