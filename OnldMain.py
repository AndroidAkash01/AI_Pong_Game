from __future__ import annotations
import sys
import time

import pygame

import config
from data_logger import DataLogger
from game_objects import Paddle, Ball, SimpleAIPaddleController
from training.inference import NeuralNetPaddleController
from training.train import train_model
from training.reinforce import ReinforcedPaddleController
from game_objects import draw_paddle, draw_ball

def ask_menu_choice() -> str:
    print("\nChoose mode:")
    print("1 = Record human gameplay data")
    print("2 = Train model")
    print("3 = Play against trained model")
    print("4 = Self-train (reinforcement)")
    print("5 = Play vs reinforced model")
    choice = input("Enter choice: ").strip()
    return choice


def ask_side(prompt: str) -> str:
    while True:
        side = input(f"{prompt} (left/right): ").strip().lower()
        if side in {"left", "right"}:
            return side
        print("Please type 'left' or 'right'.")


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    return int(raw)


def ask_hidden_sizes(default: str = "15,15") -> list[int]:
    while True:
        raw = input(f"Hidden layer sizes comma-separated [{default}]: ").strip()
        if raw == "":
            raw = default

        try:
            values = [int(part.strip()) for part in raw.split(",") if part.strip()]
            if not values or any(v <= 0 for v in values):
                raise ValueError
            return values
        except ValueError:
            print("Enter values like 30 or 15,15")


def get_model_path_for_side(side: str):
    return config.LEFT_MODEL_FILE if side == "left" else config.RIGHT_MODEL_FILE


def get_reinforced_model_path(side: str):
    if side == "left":
        return config.REINFORCED_LEFT_MODEL_FILE
    if side == "right":
        return config.REINFORCED_RIGHT_MODEL_FILE
    raise ValueError("Side must be left or right")


def draw_center_line(screen: pygame.Surface) -> None:
    dash_height = 20
    gap = 14
    x = config.SCREEN_WIDTH // 2
    for y in range(0, config.SCREEN_HEIGHT, dash_height + gap):
        pygame.draw.rect(screen, config.FG_COLOR, (x - 2, y, 4, dash_height))


def draw_text(screen: pygame.Surface, font, text: str, x: int, y: int, color) -> None:
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def create_game_objects():
    left_paddle = Paddle(
        x=config.PADDLE_MARGIN,
        y=(config.SCREEN_HEIGHT - config.PADDLE_HEIGHT) / 2,
    )
    right_paddle = Paddle(
        x=config.SCREEN_WIDTH - config.PADDLE_MARGIN - config.PADDLE_WIDTH,
        y=(config.SCREEN_HEIGHT - config.PADDLE_HEIGHT) / 2,
    )
    ball = Ball(
        x=config.SCREEN_WIDTH / 2 - config.BALL_SIZE / 2,
        y=config.SCREEN_HEIGHT / 2 - config.BALL_SIZE / 2,
    )
    ball.reset()
    return left_paddle, right_paddle, ball


def run_self_training(side: str, neurons: int):
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Self Training Mode")
    clock = pygame.time.Clock()

    left_paddle, right_paddle, ball = create_game_objects()

    model_path = get_reinforced_model_path(side)
    ai = ReinforcedPaddleController(side, model_path, hidden_size=neurons)
    opponent = SimpleAIPaddleController()

    running = True
    prev_vx = ball.vx

    save_timer = 0.0
    save_interval = 5.0
    simulation_steps = 20

    while running:
        dt = clock.tick(config.FPS) / 1000.0
        step_dt = dt / simulation_steps

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ai.save()
                print("Saved reinforced model")
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                ai.save()
                print("Saved reinforced model")
                running = False

        for _ in range(simulation_steps):
            action = ai.step(ball, left_paddle, right_paddle)

            if side == "left":
                left_paddle.move(action, step_dt)
                right_paddle.move(opponent.get_direction(right_paddle, ball), step_dt)
            else:
                right_paddle.move(action, step_dt)
                left_paddle.move(opponent.get_direction(left_paddle, ball), step_dt)

            scorer = ball.update(step_dt, left_paddle, right_paddle)
            reward = 0.0

            if prev_vx < 0 and ball.vx > 0 and side == "left":
                reward = 1.0
            elif prev_vx > 0 and ball.vx < 0 and side == "right":
                reward = 1.0

            if scorer is not None:
                paddle = left_paddle if side == "left" else right_paddle
                distance = abs(ball.center_y - paddle.center_y)
                reward = -(distance / config.SCREEN_HEIGHT) * 6.0
                ball.reset()

            next_state = ai.get_state(ball, left_paddle, right_paddle)
            ai.learn(reward, next_state)
            prev_vx = ball.vx

        save_timer += dt
        if save_timer >= save_interval:
            ai.save()
            print("Auto-saved reinforced model")
            save_timer = 0.0

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), left_paddle.rect())
        pygame.draw.rect(screen, (255, 255, 255), right_paddle.rect())
        pygame.draw.rect(screen, (255, 200, 0), ball.rect())
        pygame.display.flip()

    pygame.quit()


def build_opponent_controller(mode: str, side: str):
    if mode == "simple":
        return SimpleAIPaddleController()

    if mode == "trained":
        model_path = get_model_path_for_side(side)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        return NeuralNetPaddleController(model_path, side=side)

    if mode == "reinforced":
        model_path = get_reinforced_model_path(side)
        if not model_path.exists():
            raise FileNotFoundError(
                f"\nReinforced model not found:\n{model_path}\nRun self-training first."
            )
        return NeuralNetPaddleController(model_path, side=side)

    return SimpleAIPaddleController()


def get_ai_direction(controller, paddle, ball, left_paddle, right_paddle) -> int:
    if isinstance(controller, NeuralNetPaddleController):
        return controller.predict_direction(ball, left_paddle, right_paddle)
    return controller.get_direction(paddle, ball)


def run_game(mode: str, human_side: str, opponent_mode: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.TITLE)
    clock = pygame.time.Clock()

    font_small = pygame.font.SysFont("consolas", 22)
    font_large = pygame.font.SysFont("consolas", 36)

    left_paddle, right_paddle, ball = create_game_objects()

    opponent_side = "right" if human_side == "left" else "left"
    opponent_controller = build_opponent_controller(opponent_mode, opponent_side)

    logger = DataLogger(human_side=human_side) if mode == "record" else None

    score_left = 0
    score_right = 0

    running = True
    while running:
        dt = clock.tick(config.FPS) / 1000.0
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    ball.reset()

        _, mouse_y = pygame.mouse.get_pos()

        if human_side == "left":
            human_action = left_paddle.move_to_cursor(mouse_y, dt)
            right_dir = get_ai_direction(
                opponent_controller, right_paddle, ball, left_paddle, right_paddle
            )
            right_paddle.move(right_dir, dt)
        else:
            human_action = right_paddle.move_to_cursor(mouse_y, dt)
            left_dir = get_ai_direction(
                opponent_controller, left_paddle, ball, left_paddle, right_paddle
            )
            left_paddle.move(left_dir, dt)

        scorer = ball.update(dt, left_paddle, right_paddle)
        if scorer == "left":
            score_left += 1
            ball.reset(serve_to="right")
        elif scorer == "right":
            score_right += 1
            ball.reset(serve_to="left")

        if logger is not None and logger.should_sample(now):
            logger.record_sample(
                now=now,
                ball=ball,
                left_paddle=left_paddle,
                right_paddle=right_paddle,
                human_action=human_action,
                score_left=score_left,
                score_right=score_right,
            )

        screen.fill((10, 12, 25))  # darker base
        for i in range(0, config.SCREEN_HEIGHT, 30):
            pygame.draw.rect(
                screen,
                (100, 100, 120),
                (config.SCREEN_WIDTH // 2 - 2, i, 4, 15),
                border_radius=2
            )

        draw_paddle(screen, left_paddle, config.FG_COLOR)
        draw_paddle(screen, right_paddle, config.FG_COLOR)
        draw_ball(screen, ball)

        draw_text(
            screen,
            font_large,
            f"{score_left}        {score_right}",
            config.SCREEN_WIDTH // 2 - 70,
            20,
            config.FG_COLOR,
        )

        status = (
            f"Mode: {mode.upper()} | Human: {human_side.upper()} | "
            f"Opponent: {opponent_mode.upper()}"
        )
        draw_text(
            screen,
            font_small,
            status,
            20,
            config.SCREEN_HEIGHT - 70,
            config.ACCENT_COLOR,
        )

        helper = "Move paddle with mouse/trackpad | R=reset ball | ESC=quit"
        draw_text(
            screen,
            font_small,
            helper,
            20,
            config.SCREEN_HEIGHT - 40,
            config.FG_COLOR,
        )

        pygame.display.flip()

    pygame.quit()


def main():
    choice = ask_menu_choice()

    if choice == "1":
        human_side = ask_side("Which side do you want to play and record")
        run_game(mode="record", human_side=human_side, opponent_mode="simple")
        return

    if choice == "2":
        side = ask_side("Which side model should be trained for")
        hidden_sizes = ask_hidden_sizes("15,15")
        epochs = ask_int("Epochs", 30)
        batch_size = ask_int("Batch size", 64)

        train_model(
            side=side,
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            batch_size=batch_size,
            lr=0.001,
        )
        return

    if choice == "3":
        human_side = ask_side("Which side do you want to play")
        opponent_side = "right" if human_side == "left" else "left"
        print(f"Trained model will control: {opponent_side}")
        run_game(mode="play", human_side=human_side, opponent_mode="trained")
        return

    if choice == "4":
        side = ask_side("Which side should self-train")
        model_path = get_reinforced_model_path(side)

        if not model_path.exists():
            neurons = ask_int("Number of neurons", 30)
        else:
            neurons = 30

        run_self_training(side, neurons)
        return

    if choice == "5":
        human_side = ask_side("Which side do you want to play")
        opponent_side = "right" if human_side == "left" else "left"
        print(f"Reinforced model will control: {opponent_side}")
        run_game(mode="play", human_side=human_side, opponent_mode="reinforced")
        return

    print("Invalid choice.")


if __name__ == "__main__":
    main()