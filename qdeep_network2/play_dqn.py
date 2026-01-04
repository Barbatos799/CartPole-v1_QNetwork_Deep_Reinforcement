import gymnasium as gym
import torch
import numpy as np
from pynput import keyboard
from dqn_agent import DQN

# ---------------- TARGETS ----------------
TARGET_POINTS = {
    "a": -1.5,
    "b": -0.5,
    "c":  0.5,
    "d":  1.5
}

target_x = -0.5  # default start

# ---------------- KEYBOARD HANDLER ----------------
def on_press(key):
    global target_x
    try:
        k = key.char.lower()
        if k in TARGET_POINTS:
            new_target = TARGET_POINTS[k]
            if new_target != target_x:
                target_x = new_target
                print(f"\n🎯 Target changed to {k.upper()} → x={target_x}")
    except:
        pass


listener = keyboard.Listener(on_press=on_press)
listener.start()

# ---------------- ENV ----------------
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0] + 1
action_size = env.action_space.n

model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()


state, _ = env.reset()
state = np.append(state, target_x)

print("\nPress A / D to change target | Close window to exit\n")

try:
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()

        next_state, reward, done, truncated, _ = env.step(action)

        # console visualization
        x = next_state[0]
        bar = int((x + 2.4) / 4.8 * 50)
        target_bar = int((target_x + 2.4) / 4.8 * 50)

        line = ["-"] * 50
        if 0 <= bar < 50:
            line[bar] = "🚗"
        if 0 <= target_bar < 50:
            line[target_bar] = "🎯"

        print("".join(line), end="\r")

        state = np.append(next_state, target_x)

        # ❗ End ONLY if pole actually falls
        if done:
            print("\n❌ Pole fell — episode ended")
            break

except KeyboardInterrupt:
    print("\n🛑 User stopped simulation")

env.close()
listener.stop()
#
# state, _ = env.reset()
# state = np.append(state, target_x)
#
# print("\nPress A / B / C / D to change target\n")
#
# done = False
#
# while not done:
#     state_tensor = torch.FloatTensor(state).unsqueeze(0)
#
#     with torch.no_grad():
#         action = torch.argmax(model(state_tensor)).item()
#
#     next_state, reward, done, truncated, _ = env.step(action)
#
#     # Console visualization
#     x = next_state[0]
#     bar = int((x + 2.4) / 4.8 * 50)
#     target_bar = int((target_x + 2.4) / 4.8 * 50)
#
#     line = ["-"] * 50
#     if 0 <= bar < 50:
#         line[bar] = "🚗"
#     if 0 <= target_bar < 50:
#         line[target_bar] = "🎯"
#
#     print("".join(line), end="\r")
#
#     state = np.append(next_state, target_x)
#
#     if truncated:
#         break
# 
# env.close()
# listener.stop()
print("\nEpisode finished")

# import gymnasium as gym
# import numpy as np
# import torch
# import pygame
# from dqn_agent import DQNAgent
#
# # -------------------------------
# # Target points
# # -------------------------------
# TARGET_POINTS = {
#     pygame.K_a: -1.5,
#     pygame.K_b: -0.5,
#     pygame.K_c: 0.5,
#     pygame.K_d: 1.5
# }
#
# target_x = -0.5  # default (B)
#
# # -------------------------------
# # Environment
# # -------------------------------
# env = gym.make("CartPole-v1", render_mode="human")
#
# state_size = env.observation_space.shape[0] + 1
# action_size = env.action_space.n
#
# agent = DQNAgent(state_size, action_size)
# agent.model.load_state_dict(torch.load("dqn_cartpole.pth"))
# agent.model.eval()
#
# # -------------------------------
# # Pygame init
# # -------------------------------
# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# pygame.display.set_caption("CartPole Target Control")
#
# state, _ = env.reset()
# state = np.append(state, target_x)
#
# done = False
# clock = pygame.time.Clock()
#
# print("Press A / B / C / D to change target")
#
# while not done:
#     clock.tick(60)
#
#     # -------------------------------
#     # Keyboard handling
#     # -------------------------------
#     for event in pygame.event.get():
#         if event.type == pygame.KEYDOWN:
#             if event.key in TARGET_POINTS:
#                 target_x = TARGET_POINTS[event.key]
#                 print(f"Target changed to: {target_x}")
#
#         if event.type == pygame.QUIT:
#             done = True
#
#     # -------------------------------
#     # Agent action
#     # -------------------------------
#     action = agent.act(state, epsilon=0.0)
#
#     next_state, _, done, truncated, _ = env.step(action)
#
#     next_state = np.append(next_state, target_x)
#     state = next_state
#
#     if truncated:
#         done = True
#
# env.close()
# pygame.quit()
