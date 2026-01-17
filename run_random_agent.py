#!/usr/bin/env python3
"""Run a random agent in the PointMassContinuousEnv with rendering."""

import argparse
import time
from cleanrl_utils.pointmass_continuous_env import (
    PointMassContinuousEnv,
    ContinuousConfig,
    register_pointmass_continuous_env,
    DEFAULT_ENV_ID,
)
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def display_frame(frame, step, info, title="Point Mass Environment"):
    """Display a single frame using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(frame)
    plt.title(f"{title} - Step {step}")
    plt.xlabel(
        f"Agent: {info['agent_pos']}, Goal: {info['goal_pos']}, Distance: {info['distance_to_goal']:.3f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def run_random_agent(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 500,
    seed: int = 42,
    render_delay: float = 0.1,
    save_gif: str = None,
    show_visualization: bool = True,
):
    """Run a random agent and render the environment."""

    # Register and create environment
    register_pointmass_continuous_env()
    env = gym.make(DEFAULT_ENV_ID, render_mode="rgb_array")

    # Reset environment
    obs, info = env.reset(seed=seed)
    print(f"Environment created!")
    print(f"World size: {world_width} x {world_height}")
    print(f"Agent radius: {agent_radius}")
    print(f"Goal radius: {goal_radius}")
    print(f"Max velocity: {max_velocity}")
    print(f"Max episode steps: {max_episode_steps}")
    print(f"Initial agent position: {info['agent_pos']}")
    print(f"Initial goal position: {info['goal_pos']}")
    print(f"Initial distance to goal: {info['distance_to_goal']:.3f}")
    print("\nStarting simulation...")
    print("Press Ctrl+C to stop early\n")

    # For GIF saving
    frames = []
    if save_gif:
        import imageio.v2 as imageio

    total_reward = 0
    step = 0

    try:
        while step < max_episode_steps:
            # Sample random action
            action = env.action_space.sample()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Render
            frame = env.render()
            if save_gif:
                frames.append(frame)

            # Display frame
            if show_visualization and step % 5 == 0:  # Show every 5th frame
                display_frame(frame, step, info)

            # Print status every 50 steps
            if step % 50 == 0:
                print(
                    f"Step {step}: reward={reward:.3f}, total_reward={total_reward:.3f}, "
                    f"distance_to_goal={info['distance_to_goal']:.3f}"
                )

            # Check if episode ended
            if terminated:
                print(f"\n🎉 Goal reached in {step} steps!")
                print(f"Final reward: {reward:.3f}")
                print(f"Total reward: {total_reward:.3f}")
                break
            elif truncated:
                print(f"\n⏰ Episode truncated after {step} steps")
                print(f"Final reward: {reward:.3f}")
                print(f"Total reward: {total_reward:.3f}")
                break

            # Small delay for visualization
            if render_delay > 0:
                time.sleep(render_delay)

    except KeyboardInterrupt:
        print(f"\n⏹️  Simulation stopped by user after {step} steps")
        print(f"Total reward: {total_reward:.3f}")

    # Save GIF if requested
    if save_gif and frames:
        print(f"\n💾 Saving GIF to {save_gif}...")
        import os

        os.makedirs(os.path.dirname(save_gif), exist_ok=True) if os.path.dirname(
            save_gif
        ) else None
        imageio.mimsave(save_gif, frames, duration=render_delay)
        print(f"✅ GIF saved!")

    # Final render
    final_frame = env.render()
    print(f"\nFinal state:")
    print(f"Agent position: {info['agent_pos']}")
    print(f"Goal position: {info['goal_pos']}")
    print(f"Final distance to goal: {info['distance_to_goal']:.3f}")

    env.close()
    print("Environment closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a random agent in PointMassContinuousEnv"
    )
    parser.add_argument("--world_width", type=float, default=10.0, help="World width")
    parser.add_argument("--world_height", type=float, default=10.0, help="World height")
    parser.add_argument("--agent_radius", type=float, default=0.2, help="Agent radius")
    parser.add_argument("--goal_radius", type=float, default=0.3, help="Goal radius")
    parser.add_argument(
        "--max_velocity", type=float, default=2.0, help="Maximum velocity"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=500, help="Maximum episode steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--render_delay",
        type=float,
        default=0.1,
        help="Delay between renders (seconds)",
    )
    parser.add_argument(
        "--save_gif",
        type=str,
        default=None,
        help="Path to save GIF (e.g., out/random_agent.gif)",
    )
    parser.add_argument(
        "--no_delay", action="store_true", help="No render delay (fastest)"
    )
    parser.add_argument(
        "--no_viz", action="store_true", help="Disable visualization (text only)"
    )

    args = parser.parse_args()

    if args.no_delay:
        args.render_delay = 0.0

    run_random_agent(
        world_width=args.world_width,
        world_height=args.world_height,
        agent_radius=args.agent_radius,
        goal_radius=args.goal_radius,
        max_velocity=args.max_velocity,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        render_delay=args.render_delay,
        save_gif=args.save_gif,
        show_visualization=not args.no_viz,
    )


if __name__ == "__main__":
    main()
