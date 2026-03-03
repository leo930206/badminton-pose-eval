import os
from dataclasses import dataclass

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join("models", "pose_landmarker_lite.task")
EVENT_LOG_PATH = os.path.join("output", "action_events.jsonl")
MODEL_PRED_PATH = os.path.join("output", "model_predictions.jsonl")
TEMPLATES_DIR = os.path.join("datasets", "templates")
RAW_VIDEOS_DIR = os.path.join("datasets", "raw")


@dataclass
class Config:
    prep_bent_angle: float = 120.0
    straight_angle: float = 160.0
    perfect_angle_min: float = 165.0
    elbow_feedback_threshold: float = 150.0
    min_down_speed: float = 1.5
    min_up_speed: float = 1.2
    min_drop_speed: float = 0.6
    min_horizontal_speed: float = 1.2
    smash_min_interval_ms: int = 700
    max_straighten_frames: int = 5
    feedback_duration_sec: float = 0.8
    flash_duration_sec: float = 0.5
    wrist_shoulder_tol: float = 0.08
    max_history: int = 5
