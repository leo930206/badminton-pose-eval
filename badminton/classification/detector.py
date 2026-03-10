from collections import deque

from config import Config
from badminton.classification.context import estimate_context
from badminton.pose.tracker import angle_fast_straight


class ActionDetector:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.angle_history = deque(maxlen=config.max_history)
        self.last_action_ms = -10_000
        self.prep_active = False
        self.max_wrist_y_in_prep = 1.0

    def update(self, features):
        self.angle_history.append(features["elbow_angle"])
        context = estimate_context(features)

        straight_angle = self.config.straight_angle
        min_down_speed = self.config.min_down_speed
        if context == "defense":
            straight_angle -= 10.0
            min_down_speed *= 0.8

        if features["wrist_above_shoulder"] and features["elbow_angle"] < self.config.prep_bent_angle:
            self.prep_active = True
            self.max_wrist_y_in_prep = min(self.max_wrist_y_in_prep, features["wrist_y"])

        action = None
        cooldown_ok = (features["timestamp_ms"] - self.last_action_ms) >= self.config.smash_min_interval_ms
        fast_straight = angle_fast_straight(
            self.angle_history,
            self.config.prep_bent_angle,
            straight_angle,
            self.config.max_straighten_frames,
        )

        # ── 業界標準：速度峰值偵測（Peak Velocity Detection）──────────
        # 原理：手腕速率局部最大值 = 擊球接觸瞬間（比閾值穿越更精準）
        # just_peaked=True 表示「上一幀是速率峰值」→ 接觸剛發生
        just_peaked  = features.get("wrist_speed_just_peaked", False)
        vy_at_peak   = features.get("wrist_vy_at_peak", features["wrist_vy"])

        # 頭頂球類（殺球/高遠球/吊球）：峰值偵測 + 手腕在頭頂上方 + 準備姿勢確認
        if cooldown_ok and just_peaked and features["wrist_above_head"] and self.prep_active:
            if vy_at_peak > min_down_speed:
                action = "殺球"
            elif vy_at_peak < -self.config.min_up_speed:
                action = "高遠球"
            elif vy_at_peak > self.config.min_drop_speed:
                action = "吊球"

        # 備用：若速率峰值未觸發但 fast_straight 條件已滿足（相容性保底）
        if action is None and cooldown_ok and fast_straight and features["wrist_above_head"] and self.prep_active:
            if features["wrist_vy"] > min_down_speed * 1.3:   # 備用門檻較嚴格，減少誤觸
                action = "殺球"
            elif features["wrist_vy"] < -self.config.min_up_speed * 1.3:
                action = "高遠球"

        if action is None and cooldown_ok and features["wrist_near_shoulder"]:
            if abs(features["wrist_vx"]) > self.config.min_horizontal_speed and 130 <= features["elbow_angle"] <= 175:
                action = "平抽球"

        if action is None and cooldown_ok and features["wrist_above_shoulder"]:
            if features["wrist_vy"] > self.config.min_drop_speed and 130 <= features["elbow_angle"] <= 160:
                action = "切球"

        # 挑球：手腕在肩膀以下，快速向上揮拍（峰值偵測）
        if action is None and cooldown_ok and just_peaked and not features["wrist_above_shoulder"]:
            if vy_at_peak < -self.config.min_lift_speed:
                action = "挑球"
        # 備用：挑球閾值穿越（相容性保底）
        if action is None and cooldown_ok and not features["wrist_above_shoulder"]:
            if features["wrist_vy"] < -self.config.min_lift_speed * 1.3:
                action = "挑球"

        if action:
            self.last_action_ms = features["timestamp_ms"]
            return action, context, self.max_wrist_y_in_prep

        if not features["wrist_above_shoulder"]:
            self.prep_active = False
            self.max_wrist_y_in_prep = 1.0

        return None, context, self.max_wrist_y_in_prep
