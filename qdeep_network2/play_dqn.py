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

print("\nEpisode finished")
