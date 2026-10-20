
import os
import numpy as np
import gymnasium as gym

from agent import QLearningAgent
from utils import plot_training_rewards, save_gif


# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

SEED = 42
EPISODES = 20000

np.random.seed(SEED)

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

state_size = env.observation_space.n
action_size = env.action_space.n

agent = QLearningAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9995
)

rewards = []

print("Starting Q-learning training...")

for episode in range(EPISODES):
    state, info = env.reset(seed=SEED + episode)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.choose_action(state, training=True)

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        agent.learn(
            state, action, reward, next_state, terminated
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    agent.decay_epsilon()

    if (episode + 1) % 1000 == 0:
        recent_success = np.mean(rewards[-1000:])
        print(
            f"Episode {episode + 1}/{EPISODES} | "
            f"Success rate (last 1000): {recent_success:.2%} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

env.close()

# Save learned Q-table and training rewards
np.save("models/q_table.npy", agent.q_table)
np.save("results/training_rewards.npy", np.array(rewards))

plot_training_rewards(rewards)

# Evaluate without random exploration
eval_env = gym.make("FrozenLake-v1", is_slippery=True)
evaluation_rewards = []

for episode in range(500):
    state, info = eval_env.reset(seed=100000 + episode)
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = agent.choose_action(state, training=False)

        state, reward, terminated, truncated, info = (
            eval_env.step(action)
        )

        total_reward += reward

    evaluation_rewards.append(total_reward)

success_rate = np.mean(evaluation_rewards)

print("\nEvaluation results")
print(f"Evaluation episodes: {len(evaluation_rewards)}")
print(f"Success rate: {success_rate:.2%}")

eval_env.close()

# Record a successful episode for the GIF
gif_env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    render_mode="rgb_array"
)

successful_frames = []

for attempt in range(200):
    state, info = gif_env.reset(seed=200000 + attempt)
    frames = [gif_env.render()]
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = agent.choose_action(state, training=False)

        state, reward, terminated, truncated, info = (
            gif_env.step(action)
        )

        total_reward += reward
        frames.append(gif_env.render())

    if total_reward == 1:
        successful_frames = frames
        print(f"Successful GIF episode found: {attempt + 1}")
        break

if successful_frames:
    save_gif(successful_frames)
else:
    print(
        "No successful episode found for the GIF. "
        "Consider more training or tuning the hyperparameters."
    )

gif_env.close()

print("\nAll tasks completed.")
print("Check the models/ and results/ folders.")
