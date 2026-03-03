import math
from collections import deque

from config import Config


def calculate_angle(a, b, c) -> float:
    ba_x = a.x - b.x
    ba_y = a.y - b.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y
    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def angle_fast_straight(angle_history, low, high, max_frames) -> bool:
    if len(angle_history) < 2:
        return False
    idx_low = None
    for i in range(len(angle_history) - 1, -1, -1):
        if angle_history[i] <= low:
            idx_low = i
            break
    if idx_low is None:
        return False
    for j in range(idx_low, len(angle_history)):
        if angle_history[j] >= high and (j - idx_low) <= max_frames:
            return True
    return False


class MotionTracker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.wrist_history = deque(maxlen=config.max_history)
        self.body_history = deque(maxlen=config.max_history)
        self.prev_ts_ms = None

    def update(self, timestamp_ms, wrist, shoulder, elbow, nose, left_hip, right_hip):
        dt = None
        if self.prev_ts_ms is not None:
            dt = max(1e-6, (timestamp_ms - self.prev_ts_ms) / 1000.0)
        self.prev_ts_ms = timestamp_ms

        body_x = (left_hip.x + right_hip.x) / 2
        body_y = (left_hip.y + right_hip.y) / 2

        self.wrist_history.append((timestamp_ms, wrist.x, wrist.y))
        self.body_history.append((timestamp_ms, body_x, body_y))

        vx = vy = speed = 0.0
        bvx = bvy = body_speed = 0.0
        if dt and len(self.wrist_history) >= 2:
            _, px, py = self.wrist_history[-2]
            vx = (wrist.x - px) / dt
            vy = (wrist.y - py) / dt
            speed = math.hypot(vx, vy)
        if dt and len(self.body_history) >= 2:
            _, bx, by = self.body_history[-2]
            bvx = (body_x - bx) / dt
            bvy = (body_y - by) / dt
            body_speed = math.hypot(bvx, bvy)

        angle = calculate_angle(shoulder, elbow, wrist)
        return {
            "timestamp_ms": timestamp_ms,
            "wrist_x": wrist.x,
            "wrist_y": wrist.y,
            "wrist_vx": vx,
            "wrist_vy": vy,
            "wrist_speed": speed,
            "body_speed": body_speed,
            "elbow_angle": angle,
            "wrist_above_head": wrist.y < nose.y,
            "wrist_above_shoulder": wrist.y < shoulder.y,
            "wrist_near_shoulder": abs(wrist.y - shoulder.y) < self.config.wrist_shoulder_tol,
        }
