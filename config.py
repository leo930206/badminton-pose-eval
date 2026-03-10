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

# TrackNetV3 羽球追蹤模型
TRACKNET_PATH   = os.path.join("tracknet", "ckpts", "TrackNet_best.pt")
INPAINTNET_PATH = os.path.join("tracknet", "ckpts", "InpaintNet_best.pt")

# 球速顯示換算係數（px/s → km/h）
# TrackNet 座標為原始影片像素；此為概估值，可依拍攝距離微調
BALL_SPEED_KMH_SCALE: float = 0.035


@dataclass
class Config:
    # ── 揮拍偵測：手肘角度 ──────────────────────────────────────
    prep_bent_angle: float = 130.0        # 準備揮拍：手肘彎曲到此角度以下（原 120，放寬讓更多準備姿勢被接受）
    straight_angle: float = 158.0         # 揮拍完成：手肘伸直到此角度以上（158 更接近實際擊球點，減少過早觸發）
    perfect_angle_min: float = 165.0      # 完美伸直角度（建議回饋用）
    elbow_feedback_threshold: float = 150.0

    # ── 揮拍偵測：手腕速度（歸一化座標/秒）────────────────────────
    # 說明：速度 1.0 ≈ 一幀內手腕移動 1/30 個畫面高度（30fps）
    min_down_speed: float = 0.8           # 向下揮拍速度閾值 → 殺球（原 1.5，放寬適應非職業動作）
    min_up_speed: float = 0.7             # 向上揮拍速度閾值 → 高遠球（原 1.2）
    min_drop_speed: float = 0.35          # 向下慢速閾值 → 吊球（原 0.6）
    min_horizontal_speed: float = 0.7     # 平抽球水平速度閾值（原 1.2）
    min_lift_speed: float = 0.6           # 挑球向上速度閾值（手腕在肩膀以下快速向上）

    # ── 偵測視窗：記憶幀數 ──────────────────────────────────────
    smash_min_interval_ms: int = 500      # 兩次偵測最短間隔（ms）（500ms = 可捕捉快速來回球）
    max_straighten_frames: int = 12       # 從彎到直最多幾幀（原 5=167ms，改為 12=400ms）
    max_history: int = 15                 # 記憶幀數（原 5=167ms，改為 15=500ms，避免準備動作過長被忘記）

    # ── 其他 ──────────────────────────────────────────────────
    feedback_duration_sec: float = 0.8
    flash_duration_sec: float = 0.5
    wrist_shoulder_tol: float = 0.08
