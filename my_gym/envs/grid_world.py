import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "enemies": spaces.Box(0, size - 1, shape=(2, 2), dtype=int),
                "victims": spaces.Box(0, size - 1, shape=(2, 2), dtype=int),
            }
        )

        # Actions are "right", "up", "left", "down" ("stay" default for play)
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            # 4: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        pix_square_size = self.window_size / self.size
        self._agent_img = pygame.transform.scale(
            pygame.image.load('my_gym/sprites/tiger.png'), (pix_square_size, pix_square_size)
        )
        self._enemy_img = pygame.transform.scale(
            pygame.image.load('my_gym/sprites/enemy.png'), (pix_square_size, pix_square_size)
        )
        self._victim_img = pygame.transform.scale(
            pygame.image.load('my_gym/sprites/victim.png'), (pix_square_size, pix_square_size)
        )

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "enemies": self._enemy_locations, "victims": self._victim_locations}

    def _get_info(self):
        return {
            f"enemy_distance_{i}": np.linalg.norm(
                self._agent_location - enemy_location, ord=1
            ) for i, enemy_location in enumerate(self._enemy_locations)
        } | {
            f"victim_distance_{i}": np.linalg.norm(
                self._agent_location - victim_location, ord=1
            ) for i, victim_location in enumerate(self._victim_locations)
        }

    def reset(self, seed=None, options=None):
        # seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Sample enemy's location randomly until it is unique
        self._enemy_locations = np.array([self._agent_location, self._agent_location])
        for i, enemy_location in enumerate(self._enemy_locations):
            while self._enemy_locations.tolist().count(enemy_location.tolist()) > 1 \
                    or np.array_equal(enemy_location, self._agent_location):
                self._enemy_locations[i] = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Sample victim's location randomly until it is unique
        self._victim_locations = np.array([self._agent_location, self._agent_location])
        for i, victim_location in enumerate(self._victim_locations):
            while self._victim_locations.tolist().count(victim_location.tolist()) > 1 \
                    or victim_location.tolist() in self._enemy_locations.tolist() \
                    or np.array_equal(victim_location, self._agent_location):
                self._victim_locations[i] = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        # Make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Truncated on collision with enemy
        truncated = any(np.array_equal(x, self._agent_location) for x in self._enemy_locations)
        # Terminated on all victims killed
        terminated = self._victim_locations.size == 0
        # Check collision with victim
        mask = [not np.array_equal(x, self._agent_location) for x in self._victim_locations]

        if terminated:
            reward = 1000
        elif truncated:
            reward = -2000
        elif not all(mask):
            reward = 500
            # Remove killed victim
            self._victim_locations = self._victim_locations[mask]
        else:
            reward = -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Size of a single grid square in pixels
        pix_square_size = (self.window_size / self.size)

        # Draw enemies
        for enemy_location in self._enemy_locations:
            rect = pygame.Rect(
                pix_square_size * enemy_location,
                (pix_square_size, pix_square_size),
            )
            canvas.blit(self._enemy_img, rect)
            pygame.draw.rect(canvas, (255, 255, 255), rect, 1)

        # Draw victims
        for victim_location in self._victim_locations:
            rect = pygame.Rect(
                pix_square_size * victim_location,
                (pix_square_size, pix_square_size),
            )
            canvas.blit(self._victim_img, rect)
            pygame.draw.rect(canvas, (255, 255, 255), rect, 1)

        # Draw agent
        rect = pygame.Rect(
            pix_square_size * self._agent_location,
            (pix_square_size, pix_square_size),
        )
        canvas.blit(self._agent_img, rect)
        pygame.draw.rect(canvas, (255, 255, 255), rect, 1)

        # Add grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # Copy drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
