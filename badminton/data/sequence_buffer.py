"""
骨架序列緩衝器
用途：在分析影片時，持續儲存最近幾秒的骨架資料。
當偵測到動作時，從緩衝器取出這段序列，交給 DTW 評分。
"""

from collections import deque

# 與 extract_template.py 一致的關鍵關節定義
KEY_LANDMARKS = {
    0:  "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}


def normalize_landmarks(raw_landmarks):
    """
    把 MediaPipe 原始座標正規化為相對於身體中心的座標。
    與 extract_template.py 使用相同的演算法，確保可以互相比對。
    """
    left_hip      = raw_landmarks[23]
    right_hip     = raw_landmarks[24]
    left_shoulder = raw_landmarks[11]
    right_shoulder = raw_landmarks[12]

    cx = (left_hip.x + right_hip.x) / 2
    cy = (left_hip.y + right_hip.y) / 2

    shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_cy = (left_shoulder.y + right_shoulder.y) / 2
    torso_height = ((shoulder_cx - cx) ** 2 + (shoulder_cy - cy) ** 2) ** 0.5
    if torso_height < 1e-6:
        torso_height = 1.0

    result = {}
    for idx, name in KEY_LANDMARKS.items():
        lm = raw_landmarks[idx]
        result[name] = {
            "x": round((lm.x - cx) / torso_height, 4),
            "y": round((lm.y - cy) / torso_height, 4),
            "visibility": round(lm.visibility, 3),
        }
    return result


class SequenceBuffer:
    """
    持續緩存最近 maxlen 幀的骨架序列。
    當動作被偵測到時，呼叫 get_recent() 取出這段序列。
    """

    def __init__(self, maxlen: int = 90):
        # 90 幀 ≈ 3 秒（30fps），足以涵蓋一個完整揮拍動作
        self._buffer: deque = deque(maxlen=maxlen)

    def add(self, timestamp_ms: int, raw_landmarks) -> None:
        """接收 MediaPipe 原始骨架，正規化後存入緩衝器。"""
        normalized = normalize_landmarks(raw_landmarks)
        self._buffer.append({
            "timestamp_ms": timestamp_ms,
            "landmarks": normalized,
        })

    def get_recent(self, n_frames: int = 45) -> list:
        """
        取出最近 n_frames 幀。
        45 幀 ≈ 1.5 秒，對應一個完整的揮拍準備到擊球動作。
        """
        frames = list(self._buffer)
        return frames[-n_frames:] if len(frames) >= n_frames else frames

    def clear(self) -> None:
        self._buffer.clear()
