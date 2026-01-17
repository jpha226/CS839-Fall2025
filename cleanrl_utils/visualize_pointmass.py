import argparse
import os
from typing import Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from .pointmass_grid_env import (
    GridConfig,
    PointMassGridEnv,
    register_pointmass_grid_env,
    DEFAULT_ENV_ID,
)


def visualize_random_agent(
    width: int = 12,
    height: int = 12,
    wall_prob: float = 0.25,
    allow_diagonal: bool = False,
    seed: Optional[int] = 42,
    steps: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Run a random policy in the PointMassGridEnv and visualize the rollout.

    If save_path is provided, saves a GIF there; otherwise shows a live window using imageio's pillow viewer fallback.
    """
    register_pointmass_grid_env(
        env_id=DEFAULT_ENV_ID,
        config=GridConfig(width=width, height=height, wall_prob=wall_prob, allow_diagonal=allow_diagonal),
        render_mode="rgb_array",
        seed=seed,
    )
    env: PointMassGridEnv = gym.make(DEFAULT_ENV_ID)
    obs, info = env.reset()
    max_steps = steps if steps is not None else env.max_steps

    frames = []
    frame = env.render()
    frames.append(frame)

    for _ in range(int(max_steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if terminated or truncated:
            break

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        # imageio no longer supports 'fps'; use 'duration' (seconds per frame)
        imageio.mimsave(save_path, frames, duration=1 / 8)
        if show:
            try:
                imageio.imshow(frames[-1])
            except Exception:
                pass
    else:
        # Basic preview using imageio; for richer playback, users can pass save_path to create a GIF
        try:
            imageio.imshow(frames[-1])
        except Exception:
            pass

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a random agent in PointMassGridEnv")
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--wall_prob", type=float, default=0.25)
    parser.add_argument("--allow_diagonal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None, help="Path to save GIF, e.g., out/rollout.gif")
    parser.add_argument("--no_show", action="store_true", help="Do not attempt to show a preview window")
    args = parser.parse_args()

    visualize_random_agent(
        width=args.width,
        height=args.height,
        wall_prob=args.wall_prob,
        allow_diagonal=args.allow_diagonal,
        seed=args.seed,
        steps=args.steps,
        save_path=args.save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()


