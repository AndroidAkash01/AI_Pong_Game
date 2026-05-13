from __future__ import annotations
import torch

from training.dataset import build_feature_vector


class NeuralNetPaddleController:
    def __init__(self, model_path, side: str):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.side = side

        model_type = checkpoint.get("model_type", "supervised")

        # ---------------------------
        # 🔥 REINFORCEMENT MODEL
        # ---------------------------
        if model_type == "reinforcement":
            from training.rl_model import PongRLMLP

            self.model = PongRLMLP(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
            )

        # ---------------------------
        # 🔥 SUPERVISED MODEL
        # ---------------------------
        else:
            from training.model import PongMLP

            if "hidden_sizes" in checkpoint:
                hidden_sizes = checkpoint["hidden_sizes"]
            elif "hidden_size" in checkpoint:
                hidden_sizes = [checkpoint["hidden_size"]]
            else:
                raise KeyError("Model missing hidden size info")

            self.model = PongMLP(
                input_size=checkpoint["input_size"],
                hidden_sizes=hidden_sizes,
            )

        # 🔥 load weights AFTER correct model is created
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


    def predict_direction(self, ball, left_paddle, right_paddle) -> int:

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
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

            # 🔥 IMPORTANT DIFFERENCE
            if hasattr(out, "argmax"):
                pred = out.argmax(dim=1).item()
            else:
                pred = torch.argmax(out, dim=1).item()

        mapping = {
            0: -1,
            1: 0,
            2: 1,
        }

        return mapping[pred]