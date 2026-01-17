import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register as gym_register


@dataclass
class GridConfig:
    width: int = 10
    height: int = 10
    wall_prob: float = 0.2
    max_gen_attempts: int = 100
    reward_step: float = -0.01
    reward_goal: float = 1.0
    reward_wall: float = -0.05
    allow_diagonal: bool = False
    render_cell_size: int = 16


class PointMassGridEnv(gym.Env):
    """A simple point-mass in a randomly generated grid world.

    - Observation: continuous vector of normalized coordinates and flattened wall grid
    - Action space: Box(2,) with continuous velocity in x and y directions [-1, 1]
    - Reward: small step penalty, goal reward, small penalty if attempting to walk into wall
    - Episode ends when agent reaches goal or on time limit (width*height*4)
    - Map generation retries until solvable (BFS connectivity)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: Optional[GridConfig] = None, seed: Optional[int] = None, render_mode: Optional[str] = None):
        self.config = config or GridConfig()
        self._rng = random.Random()
        self._np_rng = np.random.RandomState()
        self._seed = None
        if seed is not None:
            self.reset_seed(seed)

        self.render_mode = render_mode
        self.width = int(self.config.width)
        self.height = int(self.config.height)
        self.max_steps = max(self.width * self.height * 4, 1)

        self.allow_diagonal = bool(self.config.allow_diagonal)
        # Use a continuous action space: agent sets velocity in x and y direction, each in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation is a single continuous vector: [ax, ay, gx, gy, walls_flat...], normalized to [0,1]
        obs_len = 4 + self.width * self.height
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        self._grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._goal_pos: Tuple[int, int] = (self.width - 1, self.height - 1)
        self._steps = 0

        self._generate_new_map()

    def reset_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng.seed(self._seed)
        self._np_rng.seed(self._seed)

    # --- Map generation ---
    def _generate_new_map(self) -> None:
        for _ in range(max(1, self.config.max_gen_attempts)):
            grid = (self._np_rng.rand(self.height, self.width) < float(self.config.wall_prob)).astype(np.int8)
            # Ensure start and goal are free
            sx, sy = 0, 0
            gx, gy = self.width - 1, self.height - 1
            grid[sy, sx] = 0
            grid[gy, gx] = 0

            if self._is_solvable(grid, (sx, sy), (gx, gy)):
                self._grid = grid
                self._agent_pos = (sx, sy)
                self._goal_pos = (gx, gy)
                return
        # Fallback to an empty grid
        self._grid = np.zeros((self.height, self.width), dtype=np.int8)
        self._agent_pos = (0, 0)
        self._goal_pos = (self.width - 1, self.height - 1)

    def _is_solvable(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        sx, sy = start
        gx, gy = goal
        if grid[sy, sx] == 1 or grid[gy, gx] == 1:
            return False
        
        # Define possible moves based on allow_diagonal setting
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        if self.allow_diagonal:
            moves += [(1, -1), (1, 1), (-1, 1), (-1, -1)]  # diagonals
        
        queue: deque = deque([(sx, sy)])
        seen = set([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            if (x, y) == (gx, gy):
                return True
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and grid[ny, nx] == 0 and (nx, ny) not in seen:
                    # If diagonal, enforce no corner cutting through walls
                    if abs(dx) + abs(dy) == 2:
                        if grid[y, nx] == 1 and grid[ny, x] == 1:
                            continue
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return False

    # --- Gym API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.reset_seed(seed)
        # Optionally regenerate a new map each episode
        regen = bool(options.get("regenerate", True)) if options else True
        if regen:
            self._generate_new_map()
        else:
            # Just reset agent to start if not regenerating
            self._agent_pos = (0, 0)
            if self._grid[self._agent_pos[1], self._agent_pos[0]] == 1:
                self._grid[self._agent_pos[1], self._agent_pos[0]] = 0
        self._steps = 0
        observation = self._get_obs()
        info: Dict = {}
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        self._steps += 1
        
        # Convert continuous action to discrete movement
        # action is a 2D vector [vx, vy] in [-1, 1]
        vx, vy = action[0], action[1]
        
        # Convert to discrete movement: round to nearest integer direction
        dx = 1 if vx > 0.5 else (-1 if vx < -0.5 else 0)
        dy = 1 if vy > 0.5 else (-1 if vy < -0.5 else 0)
        
        # If diagonal movement is not allowed, prioritize one direction
        if not self.allow_diagonal and dx != 0 and dy != 0:
            if abs(vx) > abs(vy):
                dy = 0
            else:
                dx = 0
        
        x, y = self._agent_pos
        nx, ny = x + dx, y + dy

        reward = float(self.config.reward_step)
        terminated = False
        truncated = False

        if 0 <= nx < self.width and 0 <= ny < self.height and self._grid[ny, nx] == 0:
            # prevent cutting corners with diagonals
            if abs(dx) + abs(dy) == 2:
                if self._grid[y, nx] == 1 and self._grid[ny, x] == 1:
                    reward += float(self.config.reward_wall)
                else:
                    self._agent_pos = (nx, ny)
            else:
                self._agent_pos = (nx, ny)
        else:
            reward += float(self.config.reward_wall)

        if self._agent_pos == self._goal_pos:
            reward += float(self.config.reward_goal)
            terminated = True

        if self._steps >= self.max_steps:
            truncated = True

        observation = self._get_obs()
        info: Dict = {}
        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        ax, ay = self._agent_pos
        gx, gy = self._goal_pos
        # normalize coordinates to [0,1]
        axn = 0.0 if self.width <= 1 else ax / (self.width - 1)
        ayn = 0.0 if self.height <= 1 else ay / (self.height - 1)
        gxn = 0.0 if self.width <= 1 else gx / (self.width - 1)
        gyn = 0.0 if self.height <= 1 else gy / (self.height - 1)
        walls = self._grid.astype(np.float32).reshape(-1)
        vec = np.concatenate([
            np.array([axn, ayn, gxn, gyn], dtype=np.float32),
            walls,
        ]).astype(np.float32)
        return vec

    # --- Rendering ---
    def render(self):
        if self.render_mode not in (None, "rgb_array"):
            raise ValueError("Only render_mode=None or 'rgb_array' is supported")
        return self._render_rgb()

    def _render_rgb(self) -> np.ndarray:
        cs = int(self.config.render_cell_size)
        h, w = self.height * cs, self.width * cs
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # colors
        wall_color = np.array([30, 30, 30], dtype=np.uint8)
        floor_color = np.array([220, 220, 220], dtype=np.uint8)
        agent_color = np.array([40, 120, 220], dtype=np.uint8)
        goal_color = np.array([40, 180, 80], dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                color = wall_color if self._grid[y, x] == 1 else floor_color
                yy0, yy1 = y * cs, (y + 1) * cs
                xx0, xx1 = x * cs, (x + 1) * cs
                img[yy0:yy1, xx0:xx1, :] = color

        ax, ay = self._agent_pos
        gx, gy = self._goal_pos
        ay0, ay1 = ay * cs, (ay + 1) * cs
        ax0, ax1 = ax * cs, (ax + 1) * cs
        gy0, gy1 = gy * cs, (gy + 1) * cs
        gx0, gx1 = gx * cs, (gx + 1) * cs
        img[ay0:ay1, ax0:ax1, :] = agent_color
        img[gy0:gy1, gx0:gx1, :] = goal_color

        return img

    def close(self):
        pass


def make_env(width: int = 10, height: int = 10, wall_prob: float = 0.2, allow_diagonal: bool = False, seed: Optional[int] = None, render_mode: Optional[str] = None) -> PointMassGridEnv:
    config = GridConfig(width=width, height=height, wall_prob=wall_prob, allow_diagonal=allow_diagonal)
    return PointMassGridEnv(config=config, seed=seed, render_mode=render_mode)

# --- Gymnasium registration helpers ---
DEFAULT_ENV_ID = "PointMassGrid-v0"

def register_pointmass_grid_env(env_id: str = DEFAULT_ENV_ID, **kwargs) -> None:
    """Register the PointMassGridEnv with Gymnasium.

    kwargs are passed to the env constructor via Gymnasium (e.g., config, seed, render_mode).
    Example: register_pointmass_grid_env(config=GridConfig(width=12, height=12))
    """
    # If already registered, Gymnasium will raise; we ignore by re-registering with new ID
    gym_register(
        id=env_id,
        entry_point="cleanrl_utils.pointmass_grid_env:PointMassGridEnv",
        kwargs=kwargs,
        max_episode_steps=None,
    )


