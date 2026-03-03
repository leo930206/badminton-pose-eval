"""
DTW（動態時間規整）評分模組
用途：將使用者的動作骨架序列與職業選手模板比對，計算相似度分數。

核心概念：
- DTW 允許兩段序列在時間軸上伸縮對齊，不要求等速。
- 距離越小 → 相似度越高 → 分數越高。
"""

import json
import os

import numpy as np

# 與 sequence_buffer.py 和 extract_template.py 一致的關節順序
JOINT_ORDER = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
]

# 各關節的權重（手腕和手肘對羽球動作影響最大）
JOINT_WEIGHTS = {
    "nose":            0.5,
    "left_shoulder":   1.0,
    "right_shoulder":  1.0,
    "left_elbow":      1.5,
    "right_elbow":     1.5,
    "left_wrist":      2.0,
    "right_wrist":     2.0,
    "left_hip":        0.5,
    "right_hip":       0.5,
}


def frame_to_vector(landmarks_dict: dict) -> np.ndarray:
    """將一幀的骨架字典轉成一維向量（含關節權重）。"""
    vec = []
    for name in JOINT_ORDER:
        lm = landmarks_dict.get(name, {"x": 0.0, "y": 0.0})
        weight = JOINT_WEIGHTS.get(name, 1.0)
        vec.extend([lm["x"] * weight, lm["y"] * weight])
    return np.array(vec, dtype=np.float32)


def dtw_distance(seq1: list, seq2: list) -> float:
    """
    計算兩個向量序列之間的 DTW 距離。
    時間複雜度 O(n*m)，對於 30-90 幀的序列效能足夠。
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(seq1[i - 1] - seq2[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def distance_to_score(distance: float, avg_len: float) -> float:
    """
    將 DTW 距離轉換成 0~100 的相似度分數。
    先做長度歸一化（避免長序列距離自然偏大），再映射到分數。
    """
    if avg_len <= 0:
        return 0.0
    normalized = distance / avg_len
    score = max(0.0, min(100.0, 100.0 - normalized * 25.0))
    return round(score, 1)


def _joint_differences(query_frames: list, tmpl_frames: list) -> dict:
    """
    找出使用者動作與模板在哪些關節差異最大。
    用於生成具體的改善建議（例如「右手肘角度不足」）。
    取兩段序列中間幀做比較。
    """
    if not query_frames or not tmpl_frames:
        return {}

    q_mid = query_frames[len(query_frames) // 2]["landmarks"]
    t_mid = tmpl_frames[len(tmpl_frames) // 2]["landmarks"]

    diffs = {}
    for joint in JOINT_ORDER:
        if joint in q_mid and joint in t_mid:
            q, t = q_mid[joint], t_mid[joint]
            diff = ((q["x"] - t["x"]) ** 2 + (q["y"] - t["y"]) ** 2) ** 0.5
            diffs[joint] = round(diff, 4)

    return diffs


# 根據差異最大的關節，轉成人類可讀的建議
_JOINT_ADVICE = {
    "right_wrist":    "右手腕位置需要調整，注意擊球點的高度",
    "left_wrist":     "左手輔助手臂位置需要調整",
    "right_elbow":    "右手肘延伸不足，嘗試更完整地伸展手臂",
    "left_elbow":     "左手肘位置需要調整",
    "right_shoulder": "右肩轉動幅度不足，增加身體旋轉",
    "left_shoulder":  "左肩位置需要調整，注意身體平衡",
    "nose":           "頭部位置偏差，保持視線跟球",
    "right_hip":      "右側髖部發力不足，嘗試利用腰部旋轉",
    "left_hip":       "左側髖部位置需調整",
}


def get_advice_from_diffs(joint_diffs: dict, top_n: int = 2) -> list:
    """從關節差異字典中取出最大的 top_n 個，轉成建議文字。"""
    if not joint_diffs:
        return []
    sorted_joints = sorted(joint_diffs.items(), key=lambda x: x[1], reverse=True)
    advice = []
    for joint, diff in sorted_joints[:top_n]:
        if diff > 0.05 and joint in _JOINT_ADVICE:
            advice.append(_JOINT_ADVICE[joint])
    return advice


class DTWScorer:
    """
    DTW 評分器主類別。
    - 自動載入 datasets/templates/ 下的所有模板
    - 對每個偵測到的動作進行比對，返回分數與建議
    - 若尚無模板，返回 None（不影響其他功能運作）
    """

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self._cache: dict = {}

    def _load_templates(self, action_type: str) -> list:
        """載入並快取某動作類型的所有模板。"""
        key = action_type.lower()
        if key in self._cache:
            return self._cache[key]

        action_dir = os.path.join(self.templates_dir, key)
        templates = []
        if os.path.isdir(action_dir):
            for filename in sorted(os.listdir(action_dir)):
                if filename.endswith(".json"):
                    path = os.path.join(action_dir, filename)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            templates.append(json.load(f))
                    except (json.JSONDecodeError, OSError):
                        continue

        self._cache[key] = templates
        return templates

    def score(self, action_type: str, query_frames: list):
        """
        對一個動作片段進行 DTW 評分。

        回傳：
            (score, best_template_name, advice_list)
            score: 0~100 的相似度分數，None 表示尚無模板
            best_template_name: 最相似的模板名稱
            advice_list: 具體改善建議
        """
        templates = self._load_templates(action_type)
        if not templates or not query_frames:
            return None, None, []

        query_vecs = [frame_to_vector(f["landmarks"]) for f in query_frames]

        best_score = -1.0
        best_name = None
        best_diffs = {}

        for tmpl in templates:
            tmpl_vecs = [frame_to_vector(f["landmarks"]) for f in tmpl["frames"]]
            dist = dtw_distance(query_vecs, tmpl_vecs)
            avg_len = (len(query_vecs) + len(tmpl_vecs)) / 2.0
            sc = distance_to_score(dist, avg_len)

            if sc > best_score:
                best_score = sc
                best_name = tmpl.get("name", "unknown")
                best_diffs = _joint_differences(query_frames, tmpl["frames"])

        advice = get_advice_from_diffs(best_diffs)
        return best_score, best_name, advice

    def reload(self) -> None:
        """清除快取，重新載入模板（新增模板後呼叫）。"""
        self._cache.clear()
