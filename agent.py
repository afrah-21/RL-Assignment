
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # One Q-value for each state-action pair
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, training=True):
        # Explore by selecting a random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        # Exploit the best-known action
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state, action]

        # Do not bootstrap from a terminal state
        if terminated:
            target = reward
        else:
            target = (
                reward
                + self.gamma
                * np.max(self.q_table[next_state])
            )

        self.q_table[state, action] = (
            current_q
            + self.alpha * (target - current_q)
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
