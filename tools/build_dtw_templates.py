"""
Phase 2 工具：從 ShuttleSet 抽取職業選手骨架序列，建立 DTW 評分模板庫

使用方式：
    python tools/build_dtw_templates.py --data datasets/shuttleset

輸出：datasets/templates/{球種}/*.json
每種球種抽取 5 個代表樣本，作為 DTW 比對的「職業選手標準動作」。

模板 JSON 格式（與 dtw_scorer.py 相容）：
    {
        "name": "smash_pro_01",
        "source": "ShuttleSet (KDD 2023)",
        "frames": [
            {"landmarks": {"nose": {"x": 0.5, "y": 0.3}, "left_shoulder": ..., ...}},
            ...
        ]
    }
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 球種中文名稱 → 模板資料夾英文名稱 ────────────────────────────────────────
_STROKE_FOLDER = {
    "放小球": "net_drop",
    "擋小球": "block",
    "殺球":   "smash",
    "挑球":   "lift",
    "長球":   "clear",
    "平球":   "drive",
    "切球":   "cut",
    "推球":   "push",
    "撲球":   "net_kill",
    "勾球":   "hook",
    "發短球": "short_serve",
    "發長球": "long_serve",
}

# ── COCO 17 關節索引 → DTW 評分器關節名稱 ────────────────────────────────────
# DTW 只用 9 個關節（見 dtw_scorer.py JOINT_ORDER）
_COCO_TO_DTW = {
    0:  "nose",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
}

N_JOINTS  = 17
SEQ_LEN   = 30
N_SAMPLES = 5   # 每種球種抽取幾個模板


def _bbox_normalize(joints: np.ndarray) -> np.ndarray:
    """把 (17, 2) 座標正規化到 bounding box 基準（與推論端一致）。"""
    min_xy   = joints.min(axis=0)
    max_xy   = joints.max(axis=0)
    diagonal = float(np.sqrt(((max_xy - min_xy) ** 2).sum()))
    if diagonal < 1e-6:
        return joints - min_xy
    return (joints - min_xy) / diagonal


def _pose_to_dtw_frames(pose: np.ndarray) -> list:
    """
    把 (T, 17, 2) 骨架序列轉成 DTW scorer 所需的 frames 列表。

    每個 frame：{"landmarks": {"nose": {"x": ..., "y": ...}, ...}}
    座標已做 bbox 正規化（與推論端一致）。
    """
    frames = []
    for joints in pose:
        norm = _bbox_normalize(joints)
        landmarks = {}
        for coco_idx, joint_name in _COCO_TO_DTW.items():
            x = float(norm[coco_idx, 0])
            y = float(norm[coco_idx, 1])
            landmarks[joint_name] = {"x": x, "y": y}
        frames.append({"landmarks": landmarks})
    return frames


def build_templates(
    data_dir:      str,
    output_dir:    str = "datasets/templates",
    n_samples:     int = N_SAMPLES,
    flip_bottom:   bool = True,
    use_splits:    tuple = ("train", "test"),
    seed:          int = 42,
):
    """主函式：抽取 ShuttleSet 模板並儲存。"""
    rng       = random.Random(seed)
    data_dir  = Path(data_dir)
    out_dir   = Path(output_dir)

    print(f"資料目錄：{data_dir}")
    print(f"輸出目錄：{out_dir}")
    print(f"每種球種抽取 {n_samples} 個模板\n")

    # 蒐集所有可用的 (stroke_name, player_idx, joints_file) 三元組
    samples_by_stroke: dict[str, list] = {name: [] for name in _STROKE_FOLDER}

    for split in use_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for folder_name in os.listdir(split_dir):
            folder_path = split_dir / folder_name
            if not folder_path.is_dir():
                continue

            if folder_name.startswith("Top_"):
                player_idx  = 0
                stroke_name = folder_name[4:]
            elif folder_name.startswith("Bottom_"):
                player_idx  = 1
                stroke_name = folder_name[7:]
            else:
                continue

            if stroke_name not in _STROKE_FOLDER:
                continue

            for filename in os.listdir(folder_path):
                if filename.endswith("_joints.npy"):
                    samples_by_stroke[stroke_name].append(
                        (player_idx, folder_path / filename)
                    )

    # 對每種球種：隨機抽取 n_samples 個，轉換並儲存
    total_saved = 0
    for stroke_name, samples in samples_by_stroke.items():
        if not samples:
            print(f"  ⚠️  {stroke_name}: 無資料，跳過")
            continue

        folder_en = _STROKE_FOLDER[stroke_name]
        save_dir  = out_dir / folder_en
        save_dir.mkdir(parents=True, exist_ok=True)

        chosen = rng.sample(samples, min(n_samples, len(samples)))
        saved  = 0

        for i, (player_idx, joints_path) in enumerate(chosen, start=1):
            try:
                joints = np.load(joints_path).astype(np.float32)
            except Exception as e:
                print(f"    讀取失敗 {joints_path.name}: {e}")
                continue

            if joints.ndim != 4 or joints.shape[1] < 2 or joints.shape[2] != N_JOINTS:
                continue

            T    = joints.shape[0]
            pose = joints[:, player_idx, :, :]   # (T, 17, 2)

            # Bottom 球員水平翻轉
            if flip_bottom and player_idx == 1:
                pose = pose.copy()
                pose[:, :, 0] = -pose[:, :, 0]

            # 補齊 / 截斷到 SEQ_LEN
            if T < SEQ_LEN:
                pad  = np.zeros((SEQ_LEN - T, N_JOINTS, 2), dtype=np.float32)
                pose = np.concatenate([pad, pose], axis=0)
            else:
                pose = pose[-SEQ_LEN:]

            dtw_frames = _pose_to_dtw_frames(pose)

            template = {
                "name":   f"{folder_en}_pro_{i:02d}",
                "source": "ShuttleSet (CoachAI, KDD 2023)",
                "frames": dtw_frames,
            }

            out_path = save_dir / f"{folder_en}_pro_{i:02d}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(template, f, ensure_ascii=False, separators=(",", ":"))
            saved += 1

        total_saved += saved
        print(f"  {stroke_name:6s} ({folder_en}): 已儲存 {saved} 個模板")

    print(f"\n完成！共儲存 {total_saved} 個 DTW 模板到 {out_dir}")
    print("→ 記得重啟分析程式以載入新模板（或呼叫 dtw_scorer.reload()）")


def main():
    parser = argparse.ArgumentParser(description="從 ShuttleSet 建立 DTW 評分模板庫")
    parser.add_argument("--data",     required=True,
                        help="ShuttleSet 資料目錄（含 train/ 子資料夾）")
    parser.add_argument("--output",   default="datasets/templates",
                        help="模板輸出目錄（預設 datasets/templates）")
    parser.add_argument("--samples",  type=int, default=N_SAMPLES,
                        help=f"每種球種的模板數量（預設 {N_SAMPLES}）")
    parser.add_argument("--no-flip",  action="store_true",
                        help="不翻轉 Bottom 球員的 x 座標")
    parser.add_argument("--seed",     type=int, default=42,
                        help="隨機種子（確保每次產出一致，預設 42）")
    args = parser.parse_args()

    build_templates(
        data_dir    = args.data,
        output_dir  = args.output,
        n_samples   = args.samples,
        flip_bottom = not args.no_flip,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
