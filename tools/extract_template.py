"""
Phase 2 工具：從影片片段擷取骨架序列，存成標準模板 JSON

使用方式：
    python tools/extract_template.py \
        --video datasets/raw/lin_dan_smash.mp4 \
        --action smash \
        --start 5.2 \
        --end 6.8 \
        --name lin_dan_01

輸出：datasets/templates/smash/lin_dan_01.json
"""

import argparse
import json
import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, MODEL_URL, TEMPLATES_DIR
from badminton.pose.model_loader import ensure_model

VALID_ACTIONS = ["smash", "clear", "drop", "drive", "cut", "lift"]

# 只儲存上半身關鍵關節（節省空間，DTW 比對也更精準）
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
    把原始座標轉成「相對於身體中心」的座標。
    目的：不管選手離鏡頭遠近，模板都能正常比對。

    身體中心 = 左右髖部的中間點
    縮放基準 = 肩膀到髖部的距離（軀幹長度）
    """
    left_hip  = raw_landmarks[23]
    right_hip = raw_landmarks[24]
    left_shoulder  = raw_landmarks[11]
    right_shoulder = raw_landmarks[12]

    # 身體中心
    cx = (left_hip.x + right_hip.x) / 2
    cy = (left_hip.y + right_hip.y) / 2

    # 軀幹長度（用來縮放）
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


def extract(video_path, action, start_sec, end_sec, name):
    ensure_model(MODEL_PATH, MODEL_URL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片：{video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    start_ms = int(start_sec * 1000)
    end_ms   = int(end_sec   * 1000)

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)

        if timestamp_ms < start_ms:
            frame_idx += 1
            continue
        if timestamp_ms > end_ms:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            raw = results.pose_landmarks[0]
            normalized = normalize_landmarks(raw)
            frames.append({
                "timestamp_ms": timestamp_ms - start_ms,  # 從 0 開始
                "landmarks": normalized,
            })

        frame_idx += 1

    landmarker.close()
    cap.release()

    if not frames:
        print("[警告] 這段影片沒有偵測到任何骨架，請確認時間範圍和影片內容。")
        return

    template = {
        "action": action,
        "name": name,
        "source_video": os.path.basename(video_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "frame_count": len(frames),
        "frames": frames,
    }

    out_dir = os.path.join(TEMPLATES_DIR, action)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"[完成] 儲存 {len(frames)} 幀骨架序列 → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="從影片片段擷取骨架模板")
    parser.add_argument("--video",  required=True, help="影片路徑（放在 datasets/raw/）")
    parser.add_argument("--action", required=True, choices=VALID_ACTIONS, help="動作類型")
    parser.add_argument("--start",  required=True, type=float, help="開始時間（秒）")
    parser.add_argument("--end",    required=True, type=float, help="結束時間（秒）")
    parser.add_argument("--name",   required=True, help="模板名稱（例如 lin_dan_01）")
    args = parser.parse_args()

    print(f"動作：{args.action}  |  片段：{args.start}s ~ {args.end}s  |  名稱：{args.name}")
    extract(args.video, args.action, args.start, args.end, args.name)


if __name__ == "__main__":
    main()
