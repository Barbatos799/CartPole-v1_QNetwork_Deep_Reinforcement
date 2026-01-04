import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import torch

env = gym.make("CartPole-v1", max_episode_steps=1000)

TARGET_POINTS = [-1.5, -0.5, 0.5, 1.5]

state_size = env.observation_space.shape[0] + 1  # +1 for target_x
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
memory = ReplayBuffer()

episodes = 1200
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.02

best_reward = -float("inf")

for episode in range(episodes):
    # ✅ NEW target every episode
    target_x = np.random.choice(TARGET_POINTS)

    state, _ = env.reset()
    state = np.append(state, target_x)

    total_reward = 0
    done = False

    while not done:
        action = agent.act(state, epsilon)

        next_state, _, done, truncated, _ = env.step(action)

        x = next_state[0]
        theta = next_state[2]

        # 🎯 Goal-based reward
        position_penalty = abs(x - target_x)
        x_dot = next_state[1]

        reward = (
                1.0
                - 0.5 * abs(x - target_x)
                - 2.0 * abs(theta)
                - 0.1 * abs(x_dot)
        )
        angle_penalty = abs(theta)

        reward = 1.0 - (0.5 * position_penalty) - (2.0 * angle_penalty)

        next_state = np.append(next_state, target_x)

        memory.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(memory) > batch_size:
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            agent.train(states, actions, rewards, next_states, dones)

        if truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.model.state_dict(), "dqn_cartpole.pth")

    if episode % 50 == 0:
        print(
            f"Episode {episode} | "
            f"Target: {target_x:+.2f} | "
            f"Reward: {total_reward:.2f} | "
            f"Best: {best_reward:.2f} | "
            f"Epsilon: {epsilon:.2f}"
        )

env.close()
