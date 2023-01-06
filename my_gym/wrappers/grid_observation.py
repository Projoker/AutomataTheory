import gym
import numpy as np


class FlattenGridObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = env.size
        self.observation_space = gym.spaces.Box(low=0, high=3, shape=(env.size ** 2,), dtype=int)

    def observation(self, obs):
        grid = np.zeros((self.size, self.size))

        grid[obs['agent'][0]][obs['agent'][1]] = 1
        for x, y in obs['victims']:
            grid[x][y] = 2
        for x, y in obs['enemies']:
            grid[x][y] = 3

        return grid.flatten()
