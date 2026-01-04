import gymnasium as gym
import torch
import numpy as np
from dqn_agent import DQN

# Environment with rendering
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load trained model
model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()

state, _ = env.reset()
done = False

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    state, reward, done, truncated, _ = env.step(action)

    if truncated:
        break

env.close()
