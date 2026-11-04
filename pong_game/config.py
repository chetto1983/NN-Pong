"""Shared geometry and physical units: pixels, seconds and radians."""

WIDTH, HEIGHT = 1000, 650
FPS = 60
PHYSICS_HZ = 120
PHYSICS_DT = 1.0 / PHYSICS_HZ
TOP = 7.0
BOTTOM = HEIGHT - 7.0

PADDLE_W, PADDLE_H = 14, 104
PADDLE_MARGIN = 34
PADDLE_RADIUS = PADDLE_W / 2.0
PADDLE_SEGMENT_HALF = (PADDLE_H - PADDLE_W) / 2.0
PADDLE_MIN_Y = TOP + PADDLE_H / 2.0
PADDLE_MAX_Y = BOTTOM - PADDLE_H / 2.0
BALL_R = 9
BALL_MIN_Y = TOP + BALL_R
BALL_MAX_Y = BOTTOM - BALL_R
SPAN = BALL_MAX_Y - BALL_MIN_Y

PADDLE_SPEED = 420.0
BALL_SPEED = 360.0
MIN_BALL_SPEED = 180.0
MAX_BALL_SPEED = 900.0
PADDLE_FRICTION = 0.15
# Largest angle from the horizontal after a paddle hit. Without it a hit on a
# rounded cap can leave the ball almost vertical and stall the rally.
MAX_BOUNCE_ANGLE_DEG = 60.0
CONTROL_GAIN = 60.0
SERVE_DELAY = 0.65
DEADZONE = 4.0

BG = (8, 10, 17)
FIELD = (17, 22, 33)
WHITE = (238, 242, 248)
MUTED = (120, 135, 155)
GREEN = (35, 245, 130)
YELLOW = (255, 220, 55)
CYAN = (35, 220, 245)
RED = (255, 75, 90)
PANEL = (15, 20, 31)
GRID = (34, 43, 59)

__all__ = [name for name in globals() if name.isupper()]
