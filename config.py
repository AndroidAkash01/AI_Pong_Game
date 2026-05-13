from pathlib import Path

# Screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 120
TITLE = "Pong Trainer"

# Colors
BG_COLOR = (15, 18, 30)
FG_COLOR = (235, 235, 235)
ACCENT_COLOR = (80, 200, 255)
BALL_COLOR = (255, 220, 120)
GOOD_COLOR = (120, 240, 120)

# Paddle
PADDLE_WIDTH = 16
PADDLE_HEIGHT = 110
PADDLE_SPEED = 1000.0
PADDLE_MARGIN = 30
PADDLE_RADIUS = 10
BALL_RADIUS = 8

# Ball
BALL_SIZE = 16
BALL_START_SPEED_X = 500
BALL_START_SPEED_Y = 360
BALL_MAX_SPEED_X = 700.0
BALL_MAX_SPEED_Y = 500.0
BALL_BOUNCE_SPEEDUP = 1.04

# Logging
SAMPLE_RATE_HZ = 10
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE_HZ

# Paths
# from pathlib import Path

# Base directory = where config.py is located
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

LEFT_DATA_FILE = DATA_DIR / "human_left_sessions.jsonl"
RIGHT_DATA_FILE = DATA_DIR / "human_right_sessions.jsonl"

LEFT_MODEL_FILE = MODELS_DIR / "left_model.pt"
RIGHT_MODEL_FILE = MODELS_DIR / "right_model.pt"

REINFORCED_LEFT_MODEL_FILE = MODELS_DIR / "reinforced_left_model.pt"
REINFORCED_RIGHT_MODEL_FILE = MODELS_DIR / "reinforced_right_model.pt"

# Create folders safely
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)