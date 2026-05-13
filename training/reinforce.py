from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.optim as optim

from training.rl_model import PongRLMLP
from training.dataset import build_feature_vector
import config


class ReinforcedPaddleController:
    def __init__(self, side: str, model_path, hidden_size=30):
        self.side = side
        self.model_path = model_path

        # 🔥 Model
        self.model = PongRLMLP(input_size=6, hidden_size=hidden_size)

        # 🔥 SGD (noisy, good for RL)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.criterion = nn.MSELoss()

        # 🔥 RL params
        self.gamma = 0.95  # future reward importance
        self.epsilon = 0.7  # exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.last_state = None
        self.last_action = None

        # load existing model
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded reinforced model")

    # ---------------------------
    # STATE
    # ---------------------------
    def get_state(self, ball, left_paddle, right_paddle):
        if self.side == "left":
            row = {
                "ball_x": ball.x,
                "ball_y": ball.y,
                "ball_vx": ball.vx,
                "ball_vy": ball.vy,
                "human_paddle_y": left_paddle.y,
            }
        else:
            row = {
                "ball_x": ball.x,
                "ball_y": ball.y,
                "ball_vx": ball.vx,
                "ball_vy": ball.vy,
                "human_paddle_y": right_paddle.y,
            }

        features = build_feature_vector(row, self.side)
        return torch.tensor(features, dtype=torch.float32)

    # ---------------------------
    # ACTION (epsilon-greedy)
    # ---------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()

        mapping = {0: -1, 1: 0, 2: 1}
        return mapping[action], action

    # ---------------------------
    # LEARNING (Q-learning + noise)
    # ---------------------------
    def learn(self, reward, next_state):
        if self.last_state is None:
            return

        self.model.train()

        # current Q
        q_values = self.model(self.last_state.unsqueeze(0))
        current_q = q_values[0][self.last_action]

        # future Q
        with torch.no_grad():
            next_q_values = self.model(next_state.unsqueeze(0))
            max_next_q = torch.max(next_q_values)

        # 🔥 Q-learning target
        target_q = reward + self.gamma * max_next_q

        # 🔥 Add noise to target (helps exploration)
        noise = torch.randn(1) * 0.05
        target_q = target_q + noise

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 🔥 decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------------------
    # STEP
    # ---------------------------
    def step(self, ball, left_paddle, right_paddle):
        state = self.get_state(ball, left_paddle, right_paddle)

        action, action_index = self.select_action(state)

        self.last_state = state
        self.last_action = action_index

        return action

    # ---------------------------
    # SAVE
    # ---------------------------
    def save(self):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": 6,
                "hidden_size": self.model.net[0].out_features,
                "model_type": "reinforcement",
                "side": self.side,
            },
            self.model_path,
        )