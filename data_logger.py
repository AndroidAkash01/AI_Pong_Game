from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, asdict

import config
from utils import sign, magnitude


@dataclass
class Sample:
    session_id: str
    timestamp: float
    human_side: str

    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    ball_speed: float
    ball_dir_x: int
    ball_dir_y: int

    left_paddle_y: float
    left_paddle_vy: float

    right_paddle_y: float
    right_paddle_vy: float

    human_paddle_y: float
    human_paddle_vy: float
    opponent_paddle_y: float
    opponent_paddle_vy: float

    target_action: int

    score_left: int
    score_right: int


class DataLogger:
    def __init__(self, human_side: str):
        if human_side not in {"left", "right"}:
            raise ValueError("human_side must be 'left' or 'right'")

        self.human_side = human_side
        self.session_id = str(uuid.uuid4())
        self.last_sample_time = 0.0

        self.file_path = (
            config.LEFT_DATA_FILE if human_side == "left" else config.RIGHT_DATA_FILE
        )

    def should_sample(self, now: float) -> bool:
        return (now - self.last_sample_time) >= config.SAMPLE_INTERVAL

    def record_sample(
        self,
        now: float,
        ball,
        left_paddle,
        right_paddle,
        human_action: int,
        score_left: int,
        score_right: int,
    ) -> None:
        self.last_sample_time = now

        if self.human_side == "left":
            human_paddle_y = left_paddle.y
            human_paddle_vy = left_paddle.velocity_y
            opponent_paddle_y = right_paddle.y
            opponent_paddle_vy = right_paddle.velocity_y
        else:
            human_paddle_y = right_paddle.y
            human_paddle_vy = right_paddle.velocity_y
            opponent_paddle_y = left_paddle.y
            opponent_paddle_vy = left_paddle.velocity_y

        sample = Sample(
            session_id=self.session_id,
            timestamp=now,
            human_side=self.human_side,
            ball_x=ball.x,
            ball_y=ball.y,
            ball_vx=ball.vx,
            ball_vy=ball.vy,
            ball_speed=magnitude(ball.vx, ball.vy),
            ball_dir_x=sign(ball.vx),
            ball_dir_y=sign(ball.vy),
            left_paddle_y=left_paddle.y,
            left_paddle_vy=left_paddle.velocity_y,
            right_paddle_y=right_paddle.y,
            right_paddle_vy=right_paddle.velocity_y,
            human_paddle_y=human_paddle_y,
            human_paddle_vy=human_paddle_vy,
            opponent_paddle_y=opponent_paddle_y,
            opponent_paddle_vy=opponent_paddle_vy,
            target_action=human_action,
            score_left=score_left,
            score_right=score_right,
        )

        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(sample)) + "\n")