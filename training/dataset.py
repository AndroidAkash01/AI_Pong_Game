from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

import config


def action_to_class(action: int) -> int:
    mapping = {-1: 0, 0: 1, 1: 2}
    return mapping[action]


def normalize(value: float, max_value: float) -> float:
    if max_value == 0:
        return 0.0
    return value / max_value

def build_feature_vector(row: dict, side: str):
    screen_w = float(config.SCREEN_WIDTH)
    screen_h = float(config.SCREEN_HEIGHT)
    max_ball_vx = float(config.BALL_MAX_SPEED_X)
    max_ball_vy = float(config.BALL_MAX_SPEED_Y)

    ball_x = float(row["ball_x"])
    ball_y = float(row["ball_y"])
    ball_vx = float(row["ball_vx"])
    ball_vy = float(row["ball_vy"])
    human_paddle_y = float(row["human_paddle_y"])

    vertical_offset = ball_y - human_paddle_y

    return [
        ball_x / screen_w,
        ball_y / screen_h,
        ball_vx / max_ball_vx,
        ball_vy / max_ball_vy,
        human_paddle_y / screen_h,
        vertical_offset / screen_h,
    ]



class PongDataset(Dataset):
    def __init__(self, data_file: Path, side: str):
        self.samples: List[Tuple[List[float], int]] = []

        if not data_file.exists():
            raise FileNotFoundError(
                f"\n❌ DATA FILE NOT FOUND!\n"
                f"Expected data file:\n"
                f"{data_file}\n\n"
                f"👉 Fix:\n"
                f"1. Run the game\n"
                f"2. Choose 'Record mode'\n"
                f"3. Play on the {side} side\n"
                f"4. Then train again\n"
            )

        with data_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                if row.get("human_side") != side:
                    continue

                x = build_feature_vector(row, side)
                y = action_to_class(int(row["target_action"]))
                self.samples.append((x, y))

        if not self.samples:
            raise ValueError(f"No samples found in {data_file} for side '{side}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )