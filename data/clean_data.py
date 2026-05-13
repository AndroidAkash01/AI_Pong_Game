import json
import random

INPUT_FILE = "/Users/akashgaur/Desktop/My StartUp/ServerSideBackup/html/XA2_Project/x_funSideThing/AI_Pong5/data/human_left_sessions.jsonl"     # your current dataset
OUTPUT_FILE = "/Users/akashgaur/Desktop/My StartUp/ServerSideBackup/html/XA2_Project/x_funSideThing/AI_Pong5/data/human_left_sessions.jsonl"  # new cleaned dataset

# -----------------------
# SETTINGS (tweak if needed)
# -----------------------
MAX_DISTANCE_X = 300     # ignore ball far from paddle
CENTER_THRESHOLD = 10    # when to "stay"
DROP_STAY_PROB = 0.7     # remove 70% of "stay"

# -----------------------
# HELPERS
# -----------------------
def is_ball_moving_towards_player(sample):
    if sample["human_side"] == "left":
        return sample["ball_vx"] < 0
    else:
        return sample["ball_vx"] > 0


def is_ball_close_enough(sample):
    paddle_x = 0 if sample["human_side"] == "left" else 1000  # adjust if needed
    return abs(sample["ball_x"] - paddle_x) < MAX_DISTANCE_X


def compute_correct_action(sample):
    paddle_y = sample["human_paddle_y"]
    ball_y = sample["ball_y"]

    delta = ball_y - paddle_y

    if abs(delta) < CENTER_THRESHOLD:
        return 0
    elif delta > 0:
        return 1   # move down
    else:
        return -1  # move up


# -----------------------
# MAIN CLEANING
# -----------------------
cleaned = []
stats = {"total": 0, "kept": 0, "removed": 0}

with open(INPUT_FILE, "r") as f:
    for line in f:
        stats["total"] += 1

        try:
            sample = json.loads(line)
        except:
            stats["removed"] += 1
            continue

        # 1. Remove if ball not coming towards player
        if not is_ball_moving_towards_player(sample):
            stats["removed"] += 1
            continue

        # 2. Remove if ball too far away
        if not is_ball_close_enough(sample):
            stats["removed"] += 1
            continue

        # 3. Fix action label
        correct_action = compute_correct_action(sample)

        # 4. Drop excessive "stay"
        if correct_action == 0 and random.random() < DROP_STAY_PROB:
            stats["removed"] += 1
            continue

        sample["target_action"] = correct_action

        cleaned.append(sample)
        stats["kept"] += 1


# -----------------------
# SAVE CLEAN DATA
# -----------------------
with open(OUTPUT_FILE, "w") as f:
    for sample in cleaned:
        f.write(json.dumps(sample) + "\n")


# -----------------------
# PRINT STATS
# -----------------------
print("\n--- CLEANING DONE ---")
print(f"Total samples: {stats['total']}")
print(f"Kept samples: {stats['kept']}")
print(f"Removed samples: {stats['removed']}")

# Distribution check
from collections import Counter
actions = [s["target_action"] for s in cleaned]
print("New label distribution:", Counter(actions))