"""
Phase 2 工具：用 ShuttleSet 骨架資料訓練動作分類器

使用方式：
    python tools/train_stroke_classifier.py --data datasets/shuttleset

資料格式（ShuttleSet per-file 格式）：
    datasets/shuttleset/
      train/
        Top_{球種}/    {id}_joints.npy   # shape (T, 2, 17, 2)
        Bottom_{球種}/ {id}_joints.npy
      test/
        ...（同上）

資料說明：
    - joints.npy shape (T, 2, 17, 2)：T 幀 × 2 球員 × COCO 17 關節 × (x, y)
    - player index 0 = Top 半場球員，1 = Bottom 半場球員
    - 座標已做 hip-center 正規化（雙側髖部中心 = 原點）
    - 訓練時再套 bounding box 正規化（與推論端一致）

輸出：
    models/stroke_classifier.pkl
    models/stroke_classifier_info.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 12 種球種 ─────────────────────────────────────────────────────────────────
STROKE_NAMES_ZH = [
    "放小球",   # 0  Net drop（網前放小球）
    "擋小球",   # 1  Block（網前擋）
    "殺球",     # 2  Smash
    "挑球",     # 3  Lift / Lob（把球挑高到後場）
    "長球",     # 4  Clear（高遠球）
    "平球",     # 5  Drive（平抽球）
    "切球",     # 6  Cut / Slice
    "推球",     # 7  Push
    "撲球",     # 8  Net kill（撲殺）
    "勾球",     # 9  Hook（對角網前）
    "發短球",   # 10 Short serve
    "發長球",   # 11 Long serve
]
_STROKE_TO_CLASS = {name: i for i, name in enumerate(STROKE_NAMES_ZH)}
N_CLASSES = len(STROKE_NAMES_ZH)
N_JOINTS  = 17
SEQ_LEN   = 30


def _bbox_normalize(joints: np.ndarray) -> np.ndarray:
    """
    把 (17, 2) 骨架正規化到 bounding box 基準（與 stroke_classifier.py 推論端一致）。
    原點 = bounding box 左上角，尺度 = 對角線長度。
    """
    min_xy   = joints.min(axis=0)
    max_xy   = joints.max(axis=0)
    diagonal = float(np.sqrt(((max_xy - min_xy) ** 2).sum()))
    if diagonal < 1e-6:
        return joints - min_xy
    return (joints - min_xy) / diagonal


def _extract_features(pose: np.ndarray) -> np.ndarray:
    """
    從 (SEQ_LEN, 17, 2) 骨架序列擷取特徵向量。

    包含兩部分：
    1. 位置特徵：每幀 bbox 正規化後的關節座標 → (SEQ_LEN * 17 * 2,) = 1020 維
    2. 速度特徵：相鄰幀之間的關節位移（動作流動感）→ ((SEQ_LEN-1) * 17 * 2,) = 986 維

    合計：2006 維
    速度特徵讓模型看到「怎麼動」，而不只是「在哪裡」，
    對區分看起來相似但節奏不同的球種（推球 vs 擋小球）非常有效。
    """
    # 每幀 bbox 正規化
    norm_frames = np.stack([_bbox_normalize(f) for f in pose], axis=0)  # (30, 17, 2)

    # 位置特徵
    pos_feat = norm_frames.flatten()                        # (1020,)

    # 速度特徵（相鄰幀差分）
    vel = norm_frames[1:] - norm_frames[:-1]               # (29, 17, 2)
    vel_feat = vel.flatten()                               # (986,)

    return np.concatenate([pos_feat, vel_feat])            # (2006,)


def load_split(split_dir: Path, flip_bottom: bool = True):
    """
    載入 train 或 test 分割。

    返回：
        X: (M, 2006) float32   位置 + 速度特徵
        y: (M,) int64
        class_counts: dict  各類別樣本數
    """
    X_list, y_list = [], []
    class_counts   = {i: 0 for i in range(N_CLASSES)}

    folders = sorted(os.listdir(split_dir))
    for folder_name in folders:
        folder_path = split_dir / folder_name
        if not folder_path.is_dir():
            continue

        # ── 解析資料夾名稱 ────────────────────────────────────────────
        if folder_name.startswith("Top_"):
            player_idx  = 0
            stroke_name = folder_name[4:]
        elif folder_name.startswith("Bottom_"):
            player_idx  = 1
            stroke_name = folder_name[7:]
        else:
            continue   # 跳過未知球種資料夾（如 "未知球種…"）

        if stroke_name not in _STROKE_TO_CLASS:
            continue
        class_id = _STROKE_TO_CLASS[stroke_name]

        # ── 讀取所有樣本 ──────────────────────────────────────────────
        for filename in os.listdir(folder_path):
            if not filename.endswith("_joints.npy"):
                continue

            try:
                joints = np.load(folder_path / filename).astype(np.float32)
            except Exception:
                continue

            if joints.ndim != 4 or joints.shape[1] < 2 or joints.shape[2] != N_JOINTS:
                continue

            T    = joints.shape[0]
            pose = joints[:, player_idx, :, :]   # (T, 17, 2)

            # Bottom 球員水平翻轉（讓所有人面向同一方向，提高準確率）
            if flip_bottom and player_idx == 1:
                pose = pose.copy()
                pose[:, :, 0] = -pose[:, :, 0]

            # 補齊 / 截斷到 SEQ_LEN 幀（前端補零，與 classify() 端一致）
            if T < SEQ_LEN:
                pad  = np.zeros((SEQ_LEN - T, N_JOINTS, 2), dtype=np.float32)
                pose = np.concatenate([pad, pose], axis=0)
            else:
                pose = pose[-SEQ_LEN:]   # 取最後 SEQ_LEN 幀（最接近擊球點）

            feat = _extract_features(pose)
            X_list.append(feat)
            y_list.append(class_id)
            class_counts[class_id] += 1

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
        class_counts,
    )


def train(
    data_dir:     str,
    model_out:    str = "models/stroke_classifier.pkl",
    info_out:     str = "models/stroke_classifier_info.json",
    n_estimators: int = 300,
    flip_bottom:  bool = True,
    use_rf:       bool = False,
):
    """主訓練函式。"""
    try:
        import pickle
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("[錯誤] 請先安裝 scikit-learn：pip install scikit-learn")
        return

    data_dir  = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    print(f"資料目錄：{data_dir}")

    if not train_dir.exists():
        print(f"[錯誤] 找不到 {train_dir}，請確認路徑正確")
        return

    # ── 載入資料 ──────────────────────────────────────────────────────
    print("\n載入訓練資料中...")
    X_train, y_train, tr_counts = load_split(train_dir, flip_bottom)
    print(f"訓練集：{len(X_train)} 筆")

    if test_dir.exists():
        print("載入測試資料中...")
        X_test, y_test, te_counts = load_split(test_dir, flip_bottom)
        print(f"測試集：{len(X_test)} 筆")
    else:
        X_test, y_test, te_counts = np.array([]), np.array([]), {}
        print("（無獨立測試資料夾）")

    # ── 顯示各類別樣本數 ──────────────────────────────────────────────
    print("\n各球種樣本數（訓練 / 測試）：")
    for c in range(N_CLASSES):
        tr_c = tr_counts.get(c, 0)
        te_c = te_counts.get(c, 0) if te_counts else 0
        mark = "" if tr_c > 0 else "  ← ⚠️ 無訓練資料"
        print(f"  {STROKE_NAMES_ZH[c]:6s}: 訓練 {tr_c:5d}   測試 {te_c:4d}{mark}")

    if len(X_train) == 0:
        print("[錯誤] 訓練資料為空，請確認路徑與資料格式。")
        return

    # 若無測試集，改為 80/20 自動分割
    if len(X_test) == 0:
        from sklearn.model_selection import train_test_split
        # 確保 stratify 只在有多個類別時使用
        unique_classes = np.unique(y_train)
        stratify = y_train if len(unique_classes) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=stratify
        )
        print(f"\n無獨立測試集，改用 80/20 分割：訓練 {len(X_train)}、測試 {len(X_test)}")

    # ── 訓練分類器 ────────────────────────────────────────────────────
    # 預設：HistGradientBoosting（梯度提升，比 RandomForest 更準）
    # --rf 旗標可切回 RandomForest（速度較快，方便測試）
    if use_rf:
        print(f"\n訓練隨機森林（{n_estimators} 棵決策樹）...")
        print("（CPU 多核並行，約需 1~5 分鐘）")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    else:
        print(f"\n訓練梯度提升分類器（HistGradientBoosting，max_iter={n_estimators}）...")
        print("（支援類別權重，約需 3~10 分鐘，通常比 RandomForest 準 5~15%）")
        clf = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced",
        )
    clf.fit(X_train, y_train)

    # ── 評估 ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n測試準確率：{acc * 100:.1f}%")

    present_classes = sorted(set(y_test))
    present_names   = [STROKE_NAMES_ZH[c] for c in present_classes]
    print(classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    ))

    # ── 儲存模型 ──────────────────────────────────────────────────────
    model_dir = os.path.dirname(model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"模型已儲存：{model_out}")

    info = {
        "stroke_names": STROKE_NAMES_ZH,
        "n_classes":    N_CLASSES,
        "n_joints":     N_JOINTS,
        "seq_len":      SEQ_LEN,
        "accuracy":     round(acc, 4),
        "flip_bottom":  flip_bottom,
        "source":       "ShuttleSet (CoachAI, KDD 2023)",
    }
    with open(info_out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"資訊已儲存：{info_out}")


def main():
    parser = argparse.ArgumentParser(description="用 ShuttleSet 訓練動作分類器")
    parser.add_argument("--data",    required=True,
                        help="ShuttleSet 資料目錄（含 train/ 子資料夾）")
    parser.add_argument("--model",   default="models/stroke_classifier.pkl")
    parser.add_argument("--trees",   type=int, default=300,
                        help="決策樹數量（預設 300）")
    parser.add_argument("--no-flip", action="store_true",
                        help="不翻轉 Bottom 球員的 x 座標")
    parser.add_argument("--rf", action="store_true",
                        help="使用 RandomForest（預設為 HistGradientBoosting）")
    args = parser.parse_args()

    train(
        data_dir     = args.data,
        model_out    = args.model,
        n_estimators = args.trees,
        flip_bottom  = not args.no_flip,
        use_rf       = args.rf,
    )


if __name__ == "__main__":
    main()
