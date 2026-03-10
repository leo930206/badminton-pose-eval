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
        self.body_history  = deque(maxlen=config.max_history)
        self.prev_ts_ms    = None
        # 速度峰值偵測（業界標準：wrist velocity peak ≈ 擊球接觸瞬間）
        self._speed_buf: deque = deque(maxlen=5)   # 最近 5 幀的手腕速率
        self._vy_buf:    deque = deque(maxlen=5)   # 最近 5 幀的垂直速度

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

        # ── 速度峰值偵測 ─────────────────────────────────────────────
        # 原理：擊球接觸瞬間 = 手腕合速率的局部最大值（業界 IMU 標準做法移植至視訊）
        # 實作：[s_{n-2}, s_{n-1}, s_n]，若 s_{n-1} > s_n AND s_{n-1} > s_{n-2} → 峰值剛過
        self._speed_buf.append(speed)
        self._vy_buf.append(vy)

        wrist_speed_just_peaked = False
        wrist_vy_at_peak        = vy   # 峰值那幀的垂直速度（用於判斷方向）
        if len(self._speed_buf) >= 3:
            s = list(self._speed_buf)
            if s[-2] > s[-1] and s[-2] > s[-3]:   # s[-2] 是局部最大值
                wrist_speed_just_peaked = True
                wrist_vy_at_peak = list(self._vy_buf)[-2]
        # ─────────────────────────────────────────────────────────────

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
            # 速度峰值特徵（用於精準擊球計時）
            "wrist_speed_just_peaked": wrist_speed_just_peaked,
            "wrist_vy_at_peak":        wrist_vy_at_peak,
        }

    def wrist_range_recent(self, n: int = 15) -> float:
        """近 n 幀手腕位置的包圍盒對角線長度（歸一化座標）。
        太小 → 只是輕微晃動，非真實揮拍。"""
        if len(self.wrist_history) < 2:
            return 0.0
        recent = list(self.wrist_history)[-n:]
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
