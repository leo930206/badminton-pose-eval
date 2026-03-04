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

        if cooldown_ok and fast_straight and features["wrist_above_head"] and self.prep_active:
            if features["wrist_vy"] > min_down_speed:
                action = "殺球"
            elif features["wrist_vy"] < -self.config.min_up_speed:
                action = "高遠球"
            elif features["wrist_vy"] > self.config.min_drop_speed:
                action = "吊球"

        if action is None and cooldown_ok and features["wrist_near_shoulder"]:
            if abs(features["wrist_vx"]) > self.config.min_horizontal_speed and 130 <= features["elbow_angle"] <= 175:
                action = "平抽球"

        if action is None and cooldown_ok and features["wrist_above_shoulder"]:
            if features["wrist_vy"] > self.config.min_drop_speed and 130 <= features["elbow_angle"] <= 160:
                action = "切球"

        if action:
            self.last_action_ms = features["timestamp_ms"]
            return action, context, self.max_wrist_y_in_prep

        if not features["wrist_above_shoulder"]:
            self.prep_active = False
            self.max_wrist_y_in_prep = 1.0

        return None, context, self.max_wrist_y_in_prep
