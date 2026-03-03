"""
背景分析執行緒
用途：在不阻塞 GUI 的情況下，於背景執行影片分析。
使用 PyQt5 的 QThread + 信號機制與主視窗溝通。
"""

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
from badminton.display.renderer import draw_landmarks
from badminton.pose.model_loader import ensure_model
from badminton.pose.tracker import MotionTracker
from badminton.scoring.dtw_scorer import DTWScorer
from badminton.scoring.rule_engine import grade_action
from config import (
    Config, MODEL_PATH, MODEL_URL, TEMPLATES_DIR,
)


class AnalysisWorker(QThread):
    """
    背景執行緒，負責：
    1. 讀取影片並逐幀分析
    2. 透過信號把結果傳給主視窗

    信號說明：
        frame_ready   : 每幀處理完後發出（含畫面 numpy array）
        action_found  : 偵測到動作時發出
        progress      : 進度更新 (當前幀, 總幀數)
        finished_ok   : 分析完成，附帶 event_log 和影片總時長
        error         : 發生錯誤
    """

    frame_ready  = pyqtSignal(np.ndarray)
    action_found = pyqtSignal(str, object, list)   # (action, dtw_score_or_None, advice)
    stats_updated = pyqtSignal(float, str, dict)   # (wrist_speed, context, action_counts)
    progress     = pyqtSignal(int, int)             # (current_frame, total_frames)
    finished_ok  = pyqtSignal(list, int)            # (event_log, total_ms)
    error        = pyqtSignal(str)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._analyze()
        except Exception as e:
            self.error.emit(str(e))

    def _analyze(self):
        ensure_model(MODEL_PATH, MODEL_URL)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"無法開啟影片：{self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = PoseLandmarker.create_from_options(options)

        config       = Config()
        tracker      = MotionTracker(config)
        detector     = ActionDetector(config)
        seq_buffer   = SequenceBuffer(maxlen=90)
        dtw_scorer   = DTWScorer(TEMPLATES_DIR)
        event_log    = []
        action_counts = {n: 0 for n in ["殺球", "高遠球", "吊球", "平抽球", "切球"]}
        frame_idx    = 0

        while cap.isOpened() and not self._stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            instant_speed = 0.0
            context = "neutral"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]

                # 加入序列緩衝器（供 DTW 使用）
                seq_buffer.add(timestamp_ms, landmarks)

                right_shoulder = landmarks[12]
                right_elbow    = landmarks[14]
                right_wrist    = landmarks[16]
                nose           = landmarks[0]
                left_hip       = landmarks[23]
                right_hip      = landmarks[24]

                features = tracker.update(
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

                    # DTW 比對（若模板不存在則 score=None）
                    query_seq = seq_buffer.get_recent(45)
                    dtw_score, _, dtw_advice = dtw_scorer.score(current_action, query_seq)
                    advice = dtw_advice if dtw_advice else advice_rule

                    # 存入 event_log（加上 dtw_score 欄位）
                    record = {
                        "timestamp_ms": features["timestamp_ms"],
                        "action": current_action,
                        "grade": "DTW" if dtw_score is not None else "Rule",
                        "context": context,
                        "elbow_angle": round(features["elbow_angle"], 2),
                        "wrist_speed": round(features["wrist_speed"], 3),
                        "wrist_vx": round(features["wrist_vx"], 3),
                        "wrist_vy": round(features["wrist_vy"], 3),
                        "wrist_y": round(features["wrist_y"], 3),
                        "dtw_score": dtw_score,
                        "advice": advice,
                    }
                    event_log.append(record)
                    self.action_found.emit(current_action, dtw_score, advice)

                # 繪製骨架到畫面
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

            self.stats_updated.emit(instant_speed, context, dict(action_counts))
            self.frame_ready.emit(frame)
            self.progress.emit(frame_idx, total_frames)
            frame_idx += 1

        landmarker.close()
        cap.release()

        total_ms = int(frame_idx * 1000 / fps)
        self.finished_ok.emit(event_log, total_ms)
