import gymnasium as gym
import torch
import numpy as np
import time
from dqn_agent import DQN

# ---------------- TARGETS ----------------
TARGET_POINTS = [-1.5, -0.5, 0.5, 1.5]
target_x = TARGET_POINTS[1]

# ---------------- ENV WITH VIDEO ----------------
env = gym.make(
    "CartPole-v1",
    render_mode="rgb_array"
)

env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",
    name_prefix="goal_conditioned_cartpole",
    episode_trigger=lambda e: True
)

# ---------------- MODEL ----------------
state_size = env.observation_space.shape[0] + 1
action_size = env.action_space.n

model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()

# ---------------- RUN ----------------
state, _ = env.reset()
state = np.append(state, target_x)

step = 0
done = False

while not done:
    # change target every 150 steps
    if step % 150 == 0:
        target_x = np.random.choice(TARGET_POINTS)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    next_state, reward, done, truncated, _ = env.step(action)

    state = np.append(next_state, target_x)
    step += 1

    if truncated:
        break

env.close()
print("✅ Video saved in ./videos/")
