"""
背景分析執行緒
用途：在不阻塞 GUI 的情況下，於背景執行影片分析。
使用 PyQt5 的 QThread + 信號機制與主視窗溝通。

分析分兩趟：
  Pass 1（0-50%）：TrackNetV3 偵測羽球位置
  Pass 2（50-100%）：MediaPipe 骨架分析 + 動作辨識
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, pose_landmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from PyQt5.QtCore import QThread, pyqtSignal

from badminton.classification.detector import ActionDetector
from badminton.data.logger import log_event
from badminton.data.sequence_buffer import SequenceBuffer
from badminton.display.renderer import draw_landmarks, draw_shuttle_trail
from badminton.pose.model_loader import ensure_model
from badminton.pose.tracker import MotionTracker
from badminton.scoring.dtw_scorer import DTWScorer
from badminton.scoring.rule_engine import grade_action
from badminton.tracking.shuttle_tracker import ShuttleTracker
from config import (
    Config, INPAINTNET_PATH, MODEL_PATH, MODEL_URL,
    TEMPLATES_DIR, TRACKNET_PATH,
)

_TRAIL_LEN = 12     # 殘影保留幀數


class AnalysisWorker(QThread):
    """
    背景執行緒，負責：
    1. Pass 1：以 TrackNetV3 偵測每幀羽球位置
    2. Pass 2：逐幀骨架分析 + 動作辨識
    3. 透過信號把結果傳給主視窗

    信號說明：
        frame_ready    每幀處理完後發出（含骨架 + 球軌跡的畫面）
        action_found   偵測到動作時發出（含時間戳記）
        stats_updated  每幀更新即時狀態 (wrist_speed, context, counts, ball_speed)
        progress       進度更新 (current, total)
        frame_progress Pass 2 逐幀進度 (frame_idx, total_frames)，供時間軸使用
        status_msg     狀態文字更新
        finished_ok    分析完成（event_log, total_ms, ball_positions, total_frames）
        error          發生錯誤
    """

    frame_ready    = pyqtSignal(np.ndarray)
    action_found   = pyqtSignal(str, object, list, int)   # +timestamp_ms
    stats_updated  = pyqtSignal(float, str, dict, float)
    progress       = pyqtSignal(int, int)
    frame_progress = pyqtSignal(int, int)                  # frame_idx, total_frames
    status_msg     = pyqtSignal(str)
    finished_ok    = pyqtSignal(list, int, dict, int)      # +ball_positions, total_frames
    error          = pyqtSignal(str)

    def __init__(self, video_path: str, normal_speed: bool = False, parent=None):
        super().__init__(parent)
        self.video_path   = video_path
        self.normal_speed = normal_speed
        self._stop_flag   = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._analyze()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    # ──────────────────────────────────────────

    def _analyze(self):
        ensure_model(MODEL_PATH, MODEL_URL)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"無法開啟影片：{self.video_path}")
            return

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ── Pass 1：TrackNetV3 羽球偵測（進度 0 → 50%）──
        self.status_msg.emit("正在偵測羽球軌跡（Pass 1/2）…")
        ball_positions: dict = {}

        try:
            shuttle = ShuttleTracker(TRACKNET_PATH, INPAINTNET_PATH)

            def _ball_cb(done, total):
                if not self._stop_flag:
                    self.progress.emit(done, total * 2)

            ball_positions = shuttle.track(self.video_path, progress_callback=_ball_cb)
        except Exception:
            pass    # TrackNet 失敗時以空字典繼續，不中斷整體分析

        if self._stop_flag:
            self.finished_ok.emit([], 0, {}, 0)
            return

        self.progress.emit(total_frames, total_frames * 2)

        # ── Pass 2：骨架分析（進度 50 → 100%）──
        self.status_msg.emit("正在分析動作（Pass 2/2）…")
        self.frame_progress.emit(0, total_frames)

        options    = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = PoseLandmarker.create_from_options(options)

        config        = Config()
        tracker_pose  = MotionTracker(config)
        detector      = ActionDetector(config)
        seq_buffer    = SequenceBuffer(maxlen=90)
        dtw_scorer    = DTWScorer(TEMPLATES_DIR)
        event_log     = []
        action_counts = {n: 0 for n in ["殺球", "高遠球", "吊球", "平抽球", "切球"]}
        frame_idx     = 0
        ball_trail: deque = deque(maxlen=_TRAIL_LEN)

        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened() and not self._stop_flag:
            _t0 = time.perf_counter() if self.normal_speed else 0.0

            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results      = landmarker.detect_for_video(mp_image, timestamp_ms)

            instant_speed = 0.0
            context       = "neutral"
            ball_speed    = 0.0

            # 更新球軌跡殘影
            ball_pos = ball_positions.get(frame_idx)
            ball_trail.appendleft(ball_pos)
            ball_speed = _calc_ball_speed(ball_trail, fps)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]

                seq_buffer.add(timestamp_ms, landmarks)

                right_shoulder = landmarks[12]
                right_elbow    = landmarks[14]
                right_wrist    = landmarks[16]
                nose           = landmarks[0]
                left_hip       = landmarks[23]
                right_hip      = landmarks[24]

                features = tracker_pose.update(
                    timestamp_ms, right_wrist, right_shoulder,
                    right_elbow, nose, left_hip, right_hip,
                )
                instant_speed = features["wrist_speed"]

                current_action, context, max_wrist_y = detector.update(features)

                if current_action:
                    action_counts[current_action] += 1
                    _, advice_rule = grade_action(
                        current_action, features, context, max_wrist_y, config
                    )

                    query_seq             = seq_buffer.get_recent(45)
                    dtw_score, _, dtw_adv = dtw_scorer.score(current_action, query_seq)
                    advice                = dtw_adv if dtw_adv else advice_rule

                    hit_height = _calc_hit_height(ball_pos, frame.shape[0])

                    record = {
                        "timestamp_ms": features["timestamp_ms"],
                        "action":       current_action,
                        "grade":        "DTW" if dtw_score is not None else "Rule",
                        "context":      context,
                        "elbow_angle":  round(features["elbow_angle"], 2),
                        "wrist_speed":  round(features["wrist_speed"], 3),
                        "wrist_vx":     round(features["wrist_vx"], 3),
                        "wrist_vy":     round(features["wrist_vy"], 3),
                        "wrist_y":      round(features["wrist_y"], 3),
                        "dtw_score":    dtw_score,
                        "ball_speed":   round(ball_speed, 1),
                        "hit_height":   round(hit_height, 3),
                        "advice":       advice,
                    }
                    event_log.append(record)
                    self.action_found.emit(current_action, dtw_score, advice, features["timestamp_ms"])

                # 繪製骨架
                color = (0, 255, 0)
                draw_landmarks(
                    frame, landmarks,
                    pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS,
                    color,
                )
                h, w = frame.shape[:2]
                ex = int(right_elbow.x * w)
                ey = int(right_elbow.y * h)
                cv2.putText(
                    frame,
                    f"{features['elbow_angle']:.1f}",
                    (ex + 10, ey - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )

            # 繪製球軌跡殘影
            draw_shuttle_trail(frame, list(ball_trail))

            self.stats_updated.emit(instant_speed, context, dict(action_counts), ball_speed)
            self.frame_ready.emit(frame)
            self.progress.emit(total_frames + frame_idx, total_frames * 2)
            self.frame_progress.emit(frame_idx, total_frames)
            frame_idx += 1

            # 正常速度模式：計算剩餘時間並等待
            if self.normal_speed:
                wait = 1.0 / fps - (time.perf_counter() - _t0)
                if wait > 0.001:
                    time.sleep(wait)

        landmarker.close()
        cap.release()

        total_ms = int(frame_idx * 1000 / fps)
        self.finished_ok.emit(event_log, total_ms, ball_positions, total_frames)


# ══════════════════════════════════════════════
# 輔助函式
# ══════════════════════════════════════════════

def _calc_ball_speed(trail: deque, fps: float) -> float:
    """從殘影計算即時球速（像素/秒）。"""
    visible = [p for p in trail if p is not None]
    if len(visible) < 2:
        return 0.0
    x1, y1 = visible[0]
    x2, y2 = visible[1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * fps


def _calc_hit_height(ball_pos, frame_height: int) -> float:
    """擊球高度比例（0 = 底部，1 = 頂部）。"""
    if ball_pos is None:
        return 0.0
    _, y = ball_pos
    return max(0.0, min(1.0, 1.0 - y / frame_height))
