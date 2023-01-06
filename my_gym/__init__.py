from gym.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="my_gym.envs:GridWorldEnv",
)
