"""
動作分類器：載入訓練好的隨機森林模型，對 MediaPipe 骨架序列做球種分類。

使用前：先執行 tools/train_stroke_classifier.py 訓練模型。
"""

import json
import os
import pickle
from collections import deque

import numpy as np

# MediaPipe 33 關節 → COCO 17 關節 索引映射
# MediaPipe landmark index → COCO joint index
_MP_TO_COCO = {
    0:  0,   # nose
    2:  1,   # left_eye
    5:  2,   # right_eye
    7:  3,   # left_ear
    8:  4,   # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}
# 依 COCO index 排序後，對應的 MediaPipe index 列表
_COCO_TO_MP = [_MP_TO_COCO[k] for k in sorted(_MP_TO_COCO.keys(),
               key=lambda mp: _MP_TO_COCO[mp])]

N_JOINTS  = 17
SEQ_LEN   = 30
# 特徵維度：位置 (30×17×2=1020) + 速度 (29×17×2=986) = 2006
FEAT_DIM  = SEQ_LEN * N_JOINTS * 2 + (SEQ_LEN - 1) * N_JOINTS * 2

# 預設球種名稱（找不到 _info.json 時使用）
DEFAULT_STROKE_NAMES = [
    "放小球", "擋小球", "殺球", "挑球", "長球", "平球",
    "切球", "推球", "撲球", "勾球", "發短球", "發長球",
]


def _mediapipe_to_coco(landmarks) -> np.ndarray:
    """
    把 MediaPipe PoseLandmarker 輸出的 33 個關節，
    轉成 COCO 格式的 17 個關節。

    輸入：MediaPipe landmarks list（33 個 NormalizedLandmark）
    輸出：np.ndarray shape (17, 2)，x/y 歸一化座標
    """
    coco = np.zeros((N_JOINTS, 2), dtype=np.float32)
    for coco_idx, mp_idx in enumerate(_COCO_TO_MP):
        lm = landmarks[mp_idx]
        coco[coco_idx, 0] = lm.x
        coco[coco_idx, 1] = lm.y
    return coco


def _bbox_normalize(joints: np.ndarray) -> np.ndarray:
    """
    把骨架座標正規化到以 bounding box 為基準（與 ShuttleSet 訓練資料一致）。

    bounding box = 所有關節的最小/最大範圍
    原點移到 bounding box 左上角，再除以對角線長度。

    輸入：(17, 2) 的絕對歸一化座標
    輸出：(17, 2) 的 bounding box 正規化座標
    """
    min_xy = joints.min(axis=0)   # (2,)
    max_xy = joints.max(axis=0)   # (2,)
    diagonal = np.sqrt(((max_xy - min_xy) ** 2).sum())
    if diagonal < 1e-6:
        return joints - min_xy
    return (joints - min_xy) / diagonal


class StrokeClassifier:
    """
    即時動作分類器。

    使用方式：
        clf = StrokeClassifier("models/stroke_classifier.pkl")

        # 每幀呼叫 add_frame：
        clf.add_frame(mediapipe_landmarks)

        # 偵測到擊球事件時呼叫 classify：
        stroke, confidence = clf.classify()
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"找不到分類器模型：{model_path}\n"
                "請先執行：python tools/train_stroke_classifier.py --data datasets/shuttleset"
            )

        with open(model_path, "rb") as f:
            self._clf = pickle.load(f)

        # 讀取類別名稱
        info_path = model_path.replace(".pkl", "_info.json")
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            self._stroke_names = info["stroke_names"]
        else:
            # 預設名稱
            self._stroke_names = DEFAULT_STROKE_NAMES

        # 滾動幀緩衝區：存最近 SEQ_LEN 幀的 COCO 骨架
        self._buffer: deque = deque(maxlen=SEQ_LEN)

    @property
    def stroke_names(self) -> list[str]:
        return self._stroke_names

    def add_frame(self, landmarks) -> None:
        """
        每幀呼叫，把 MediaPipe landmarks 加入緩衝區。

        landmarks：MediaPipe PoseLandmarker 的 pose_landmarks[0]
        """
        coco = _mediapipe_to_coco(landmarks)      # (17, 2)
        norm = _bbox_normalize(coco)              # bounding box 正規化
        self._buffer.append(norm)

    def classify(self) -> tuple[str | None, float]:
        """
        用目前緩衝區的骨架序列分類球種。

        返回：(stroke_name, confidence)
              若緩衝區幀數不足，返回 (None, 0.0)
        """
        if len(self._buffer) < SEQ_LEN // 2:   # 至少要有半段才分類
            return None, 0.0

        # 補齊到 SEQ_LEN 幀（前面補零，與訓練端一致）
        frames = list(self._buffer)
        if len(frames) < SEQ_LEN:
            pad = [np.zeros((N_JOINTS, 2), dtype=np.float32)] * (SEQ_LEN - len(frames))
            frames = pad + frames

        seq = np.stack(frames, axis=0)          # (30, 17, 2)

        # 位置特徵 + 速度特徵（與訓練端 _extract_features 一致）
        pos_feat = seq.flatten()                # (1020,)
        vel_feat = (seq[1:] - seq[:-1]).flatten()  # (986,)
        feat = np.concatenate([pos_feat, vel_feat]).reshape(1, -1)  # (1, 2006)

        proba = self._clf.predict_proba(feat)[0]   # (n_classes,)
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        stroke = self._stroke_names[pred_idx]

        return stroke, confidence

    def clear(self) -> None:
        """清空緩衝區（每次分析開始時呼叫）。"""
        self._buffer.clear()
