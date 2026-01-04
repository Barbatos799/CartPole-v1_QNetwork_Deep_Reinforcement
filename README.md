# 🎯 Goal-Conditioned CartPole (Deep Reinforcement Learning)

This project implements a **goal-conditioned Deep Q-Network (DQN)** agent on the CartPole environment.

The agent can:
- Balance the pole indefinitely
- Move to different target positions (A / D control)
- Switch goals dynamically during execution
- Record demo videos automatically

## 🚀 Features
- DQN implemented in PyTorch
- Replay Buffer
- Goal-conditioned state
- Live target switching
- Infinite-horizon control
- Video recording support

## 📦 Installation

```bash
pip install -r requirements.txt


🏃 Training
python train_dqn.py

🎮 Play (Interactive)
python play_dqn.py

🎥 Record Demo
python record_demo.py

🧠 Environment

CartPole-v1 (Gymnasium)