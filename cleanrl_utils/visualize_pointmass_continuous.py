import argparse
import os
from typing import Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from .pointmass_continuous_env import (
    ContinuousConfig,
    PointMassContinuousEnv,
    register_pointmass_continuous_env,
    DEFAULT_ENV_ID,
)


def visualize_random_agent(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 500,
    seed: Optional[int] = 42,
    steps: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Run a random policy in the PointMassContinuousEnv and visualize the rollout.

    If save_path is provided, saves a GIF there; otherwise shows a live window using imageio's pillow viewer fallback.
    """
    register_pointmass_continuous_env(
        env_id=DEFAULT_ENV_ID,
        config=ContinuousConfig(
            world_width=world_width,
            world_height=world_height,
            agent_radius=agent_radius,
            goal_radius=goal_radius,
            max_velocity=max_velocity,
            max_episode_steps=max_episode_steps
        ),
        render_mode="rgb_array",
        seed=seed,
    )
    env: PointMassContinuousEnv = gym.make(DEFAULT_ENV_ID)
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


def visualize_goal_seeking_agent(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 500,
    seed: Optional[int] = 42,
    steps: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Run a simple goal-seeking policy in the PointMassContinuousEnv and visualize the rollout."""
    register_pointmass_continuous_env(
        env_id=DEFAULT_ENV_ID,
        config=ContinuousConfig(
            world_width=world_width,
            world_height=world_height,
            agent_radius=agent_radius,
            goal_radius=goal_radius,
            max_velocity=max_velocity,
            max_episode_steps=max_episode_steps
        ),
        render_mode="rgb_array",
        seed=seed,
    )
    env: PointMassContinuousEnv = gym.make(DEFAULT_ENV_ID)
    obs, info = env.reset()
    max_steps = steps if steps is not None else env.max_steps

    frames = []
    frame = env.render()
    frames.append(frame)

    for _ in range(int(max_steps)):
        # Simple goal-seeking policy: move towards goal
        agent_pos = info["agent_pos"]
        goal_pos = info["goal_pos"]
        direction = goal_pos - agent_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        action = direction * max_velocity * 0.5  # Scale down for smoother movement
        
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if terminated or truncated:
            break

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        imageio.mimsave(save_path, frames, duration=1 / 8)
        if show:
            try:
                imageio.imshow(frames[-1])
            except Exception:
                pass
    else:
        try:
            imageio.imshow(frames[-1])
        except Exception:
            pass

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize agents in PointMassContinuousEnv")
    parser.add_argument("--world_width", type=float, default=10.0)
    parser.add_argument("--world_height", type=float, default=10.0)
    parser.add_argument("--agent_radius", type=float, default=0.2)
    parser.add_argument("--goal_radius", type=float, default=0.3)
    parser.add_argument("--max_velocity", type=float, default=2.0)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None, help="Path to save GIF, e.g., out/rollout.gif")
    parser.add_argument("--no_show", action="store_true", help="Do not attempt to show a preview window")
    parser.add_argument("--policy", type=str, choices=["random", "goal_seeking"], default="random", 
                       help="Policy to use: random or goal_seeking")
    args = parser.parse_args()

    if args.policy == "random":
        visualize_random_agent(
            world_width=args.world_width,
            world_height=args.world_height,
            agent_radius=args.agent_radius,
            goal_radius=args.goal_radius,
            max_velocity=args.max_velocity,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            steps=args.steps,
            save_path=args.save_path,
            show=not args.no_show,
        )
    elif args.policy == "goal_seeking":
        visualize_goal_seeking_agent(
            world_width=args.world_width,
            world_height=args.world_height,
            agent_radius=args.agent_radius,
            goal_radius=args.goal_radius,
            max_velocity=args.max_velocity,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            steps=args.steps,
            save_path=args.save_path,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
