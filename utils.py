from __future__ import annotations
import math


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def magnitude(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)