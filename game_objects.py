from __future__ import annotations
import random
from dataclasses import dataclass
import pygame

import config


# =========================
# PADDLE
# =========================
@dataclass
class Paddle:
    x: float
    y: float
    width: int = config.PADDLE_WIDTH
    height: int = config.PADDLE_HEIGHT
    speed: float = config.PADDLE_SPEED
    velocity_y: float = 0.0

    # 🔥 NEW: smooth rendering
    display_y: float = 0.0

    def __post_init__(self):
        self.display_y = self.y

    def move(self, direction: int, dt: float) -> None:
        old_y = self.y

        self.velocity_y = direction * self.speed
        self.y += self.velocity_y * dt

        self.clamp()

        if dt > 0:
            self.velocity_y = (self.y - old_y) / dt

        # 🔥 Smooth interpolation
        self.display_y += (self.y - self.display_y) * 0.25

    def move_to_cursor(self, mouse_y: float, dt: float) -> int:
        target_center = mouse_y
        current_center = self.center_y
        diff = target_center - current_center

        if abs(diff) < 10:
            self.velocity_y = 0.0
            return 0

        max_move = self.speed * dt
        move_amount = max(-max_move, min(max_move, diff))

        old_y = self.y
        self.y += move_amount
        self.clamp()

        if dt > 0:
            self.velocity_y = (self.y - old_y) / dt
        else:
            self.velocity_y = 0.0

        # 🔥 Smooth interpolation
        self.display_y += (self.y - self.display_y) * 0.25

        if self.velocity_y < -1:
            return -1
        if self.velocity_y > 1:
            return 1
        return 0

    def clamp(self) -> None:
        self.y = max(0, min(config.SCREEN_HEIGHT - self.height, self.y))

    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x),
            int(self.display_y),  # 🔥 use smooth value
            self.width,
            self.height
        )

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


# =========================
# BALL
# =========================
@dataclass
class Ball:
    x: float
    y: float
    size: int = config.BALL_SIZE
    vx: float = config.BALL_START_SPEED_X
    vy: float = config.BALL_START_SPEED_Y

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.size, self.size)

    @property
    def center_x(self) -> float:
        return self.x + self.size / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.size / 2.0

    def reset(self, serve_to: str | None = None) -> None:
        self.x = config.SCREEN_WIDTH / 2 - self.size / 2
        self.y = config.SCREEN_HEIGHT / 2 - self.size / 2

        direction = random.choice([-1, 1])
        if serve_to == "left":
            direction = -1
        elif serve_to == "right":
            direction = 1

        self.vx = config.BALL_START_SPEED_X * direction
        self.vy = random.choice([-1, 1]) * config.BALL_START_SPEED_Y

    def update(self, dt: float, left_paddle: Paddle, right_paddle: Paddle) -> str | None:
        self.x += self.vx * dt
        self.y += self.vy * dt

        # wall collision
        if self.y <= 0:
            self.y = 0
            self.vy *= -1
        elif self.y + self.size >= config.SCREEN_HEIGHT:
            self.y = config.SCREEN_HEIGHT - self.size
            self.vy *= -1

        ball_rect = self.rect()
        left_rect = left_paddle.rect()
        right_rect = right_paddle.rect()

        # paddle collision
        if self.vx < 0 and ball_rect.colliderect(left_rect):
            self.x = left_rect.right
            self.vx = abs(self.vx) * config.BALL_BOUNCE_SPEEDUP
            self.vy = self._calculate_bounce(left_paddle)
            self._clamp_speed()

        elif self.vx > 0 and ball_rect.colliderect(right_rect):
            self.x = right_rect.left - self.size
            self.vx = -abs(self.vx) * config.BALL_BOUNCE_SPEEDUP
            self.vy = self._calculate_bounce(right_paddle)
            self._clamp_speed()

        # scoring
        if self.x + self.size < 0:
            return "right"
        if self.x > config.SCREEN_WIDTH:
            return "left"

        return None

    def _calculate_bounce(self, paddle: Paddle) -> float:
        offset = (self.center_y - paddle.center_y) / (paddle.height / 2.0)
        offset = max(-1.0, min(1.0, offset))
        return offset * config.BALL_MAX_SPEED_Y

    def _clamp_speed(self) -> None:
        self.vx = max(-config.BALL_MAX_SPEED_X, min(config.BALL_MAX_SPEED_X, self.vx))
        self.vy = max(-config.BALL_MAX_SPEED_Y, min(config.BALL_MAX_SPEED_Y, self.vy))


# =========================
# SIMPLE AI
# =========================
class SimpleAIPaddleController:
    def __init__(self, deadzone: float = 12.0):
        self.deadzone = deadzone

    def get_direction(self, paddle: Paddle, ball: Ball) -> int:
        if ball.center_y < paddle.center_y - self.deadzone:
            return -1
        if ball.center_y > paddle.center_y + self.deadzone:
            return 1
        return 0


# =========================
# DRAW HELPERS (NEW UI)
# =========================
def draw_paddle(surface, paddle, color):
    pygame.draw.rect(
        surface,
        color,
        paddle.rect(),
        border_radius=12  # 🔥 rounded paddle
    )


def draw_ball(surface, ball):
    x = int(ball.center_x)
    y = int(ball.center_y)

    # 🔥 glow effect
    for r in range(14, 8, -2):
        glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 220, 120, 25), (r, r), r)
        surface.blit(glow, (x - r, y - r))

    pygame.draw.circle(
        surface,
        config.BALL_COLOR,
        (x, y),
        8
    )