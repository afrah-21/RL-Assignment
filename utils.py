
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def plot_training_rewards(rewards, window=100, save_path="results/training.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.25, label="Episode reward")

    if len(rewards) >= window:
        averages = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            np.arange(window - 1, len(rewards)),
            averages,
            label=f"{window}-episode moving average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("FrozenLake Q-Learning Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_gif(frames, save_path="results/trained_agent.gif"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if frames:
        imageio.mimsave(
            save_path,
            frames,
            duration=0.25,
            loop=0
        )
        print(f"GIF saved to: {save_path}")
    else:
        print("No frames were available to save.")
