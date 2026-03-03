"""
羽球軌跡追蹤器
基於 TrackNetV3（TrackNet + InpaintNet），自動使用 CUDA/CPU。

用法：
    tracker = ShuttleTracker(tracknet_path, inpaintnet_path)
    ball_pos = tracker.track(video_path, progress_callback=lambda done, total: ...)
    # ball_pos: {frame_idx: (x, y)}  座標為原始影片像素
"""

import math
import os
import sys

import cv2
import numpy as np
import torch

# ── TrackNetV3 模型空間尺寸（固定）──
HEIGHT = 288
WIDTH  = 512
DELTA_T = 1.0 / math.sqrt(HEIGHT ** 2 + WIDTH ** 2)
COOR_TH = DELTA_T * 50          # InpaintNet 座標閾值

_TRACKNET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'tracknet')
)


# ══════════════════════════════════════════════
# 從 tracknet/model.py 動態載入（避免污染 sys.modules）
# ══════════════════════════════════════════════

def _import_tracknet_models():
    sys.path.insert(0, _TRACKNET_DIR)
    try:
        from model import TrackNet, InpaintNet
    finally:
        sys.path.pop(0)
    return TrackNet, InpaintNet


def _build_tracknet(seq_len: int, bg_mode: str):
    TrackNet, _ = _import_tracknet_models()
    if bg_mode == 'subtract':
        return TrackNet(in_dim=seq_len, out_dim=seq_len)
    if bg_mode == 'subtract_concat':
        return TrackNet(in_dim=seq_len * 4, out_dim=seq_len)
    if bg_mode == 'concat':
        return TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
    return TrackNet(in_dim=seq_len * 3, out_dim=seq_len)


def _build_inpaintnet():
    _, InpaintNet = _import_tracknet_models()
    return InpaintNet()


# ══════════════════════════════════════════════
# 從 tracknet/test.py 複製的必要函式
# （避免 import test 帶入 pycocotools 依賴）
# ══════════════════════════════════════════════

def _predict_location(heatmap: np.ndarray):
    """從二值熱力圖中找出球的邊界框 (x, y, w, h)。"""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(
        heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rects = [cv2.boundingRect(c) for c in cnts]
    return max(rects, key=lambda r: r[2] * r[3])


def _generate_inpaint_mask(pred_dict: dict, th_h: float = 30.0):
    """生成 InpaintNet 遮罩：標記球消失但應當被補全的幀。"""
    y   = np.array(pred_dict['Y'])
    vis = np.array(pred_dict['Visibility'])
    mask = np.zeros_like(y)
    i = j = 0
    while j < len(vis):
        while i < len(vis) - 1 and vis[i] == 1:
            i += 1
        j = i
        while j < len(vis) - 1 and vis[j] == 0:
            j += 1
        if j == i:
            break
        if i == 0 and y[j] > th_h:
            mask[:j] = 1
        elif (i > 1 and y[i - 1] > th_h) and (j < len(vis) and y[j] > th_h):
            mask[i:j] = 1
        i = j
    return mask.tolist()


# ══════════════════════════════════════════════
# 主類別
# ══════════════════════════════════════════════

class ShuttleTracker:
    """
    羽球追蹤器（TrackNet + InpaintNet）。

    模型在 __init__ 時載入一次；track() 可多次呼叫。
    """

    def __init__(self, tracknet_path: str, inpaintnet_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # ── 載入 TrackNet ──
        ckpt = torch.load(tracknet_path, map_location=device, weights_only=False)
        self._seq_len = ckpt['param_dict']['seq_len']
        self._bg_mode = ckpt['param_dict']['bg_mode']
        self._tracknet = _build_tracknet(self._seq_len, self._bg_mode).to(device)
        self._tracknet.load_state_dict(ckpt['model'])
        self._tracknet.eval()

        # ── 載入 InpaintNet ──
        ckpt2 = torch.load(inpaintnet_path, map_location=device, weights_only=False)
        self._inpaint_seq_len = ckpt2['param_dict']['seq_len']
        self._inpaintnet = _build_inpaintnet().to(device)
        self._inpaintnet.load_state_dict(ckpt2['model'])
        self._inpaintnet.eval()

    # ── 內部工具 ──────────────────────────────

    @staticmethod
    def _read_frames(video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def _compute_median(frames_rgb_small: list, max_samples: int = 300) -> np.ndarray:
        n = len(frames_rgb_small)
        idxs = np.linspace(0, n - 1, min(max_samples, n), dtype=int)
        samples = [
            frames_rgb_small[i].astype(np.float32) / 255.0
            for i in idxs
        ]
        return np.median(np.array(samples), axis=0).astype(np.float32)

    def _make_input(
        self,
        frames_small: list,
        median: np.ndarray,
        start: int,
    ) -> torch.Tensor:
        """製作 TrackNet 的單序列輸入張量 (C, H, W)。"""
        seq = []
        for i in range(start, start + self._seq_len):
            if i < len(frames_small):
                f = frames_small[i].astype(np.float32) / 255.0
            else:
                f = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
            seq.append(f)

        if self._bg_mode == 'concat':
            parts = [median] + seq                          # (H,W,3) × (seq+1)
        elif self._bg_mode == 'subtract':
            diffs, prev = [], seq[0]
            for f in seq:
                d = np.abs(f - prev).mean(axis=2, keepdims=True)
                diffs.append(d)
                prev = f
            parts = diffs
        elif self._bg_mode == 'subtract_concat':
            diffs, prev = [], seq[0]
            for f in seq:
                d = np.abs(f - prev).mean(axis=2, keepdims=True)
                diffs.append(np.concatenate([f, d], axis=2))
                prev = f
            parts = diffs
        else:
            parts = seq

        x = np.concatenate(parts, axis=2)       # (H, W, C)
        x = np.transpose(x, (2, 0, 1))          # (C, H, W)
        return torch.from_numpy(x)

    # ── 主流程 ────────────────────────────────

    def track(self, video_path: str, progress_callback=None) -> dict:
        """
        對影片執行全幀羽球追蹤。

        Args:
            video_path:        影片路徑（mp4 / avi / mov / mkv）
            progress_callback: (done_frames, total_frames) → None

        Returns:
            {frame_idx: (x, y)}  僅包含可見幀；座標為原始影片像素。
        """
        # ── 讀取影片 ──
        frames_bgr = self._read_frames(video_path)
        if not frames_bgr:
            return {}

        h_orig, w_orig = frames_bgr[0].shape[:2]
        w_scaler = w_orig / WIDTH
        h_scaler = h_orig / HEIGHT
        total = len(frames_bgr)

        # BGR → RGB + 縮小至模型尺寸
        frames_small = [
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))
            for f in frames_bgr
        ]
        median = (
            self._compute_median(frames_small)
            if self._bg_mode == 'concat' else None
        )

        # ── Pass 1：TrackNet ──────────────────
        batch_size = 16
        seq_len    = self._seq_len
        starts     = list(range(0, total, seq_len))

        tp: dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': []}

        for b0 in range(0, len(starts), batch_size):
            b1 = min(b0 + batch_size, len(starts))
            tensors = [
                self._make_input(frames_small, median, s)
                for s in starts[b0:b1]
            ]
            x_batch = torch.stack(tensors).float().to(self.device)

            with torch.no_grad():
                y_pred = self._tracknet(x_batch).cpu()      # (N, L, H, W)

            y_bin = (y_pred > 0.5).numpy().astype(np.uint8) * 255

            for bi, s in enumerate(starts[b0:b1]):
                for fi in range(seq_len):
                    fidx = s + fi
                    if fidx >= total:
                        break
                    hm = y_bin[bi, fi]
                    rx, ry, rw, rh = _predict_location(hm)
                    cx = int((rx + rw / 2) * w_scaler)
                    cy = int((ry + rh / 2) * h_scaler)
                    vis = 0 if (cx == 0 and cy == 0) else 1
                    tp['Frame'].append(fidx)
                    tp['X'].append(cx)
                    tp['Y'].append(cy)
                    tp['Visibility'].append(vis)

            if progress_callback:
                progress_callback(min(b1 * seq_len, total), total)

        # ── Pass 2：InpaintNet ────────────────
        tp['Inpaint_Mask'] = _generate_inpaint_mask(tp, th_h=h_orig * 0.05)
        final = self._run_inpaintnet(tp, w_orig, h_orig)

        # ── 建立結果字典 ──
        return {
            f: (x, y)
            for f, x, y, vis in zip(
                final['Frame'], final['X'], final['Y'], final['Visibility']
            )
            if vis
        }

    def _run_inpaintnet(self, tp: dict, orig_w: int, orig_h: int) -> dict:
        """InpaintNet：補全遺失幀的球位置。"""
        mask = tp['Inpaint_Mask']
        if sum(mask) == 0:
            return tp                           # 無需補全

        n = len(tp['Frame'])
        seq_len = self._inpaint_seq_len

        # 正規化座標到 [0, 1]（以原始影片尺寸）
        x_norm = [x / orig_w for x in tp['X']]
        y_norm = [y / orig_h for y in tp['Y']]

        result = {
            'Frame':      list(tp['Frame']),
            'X':          list(tp['X']),
            'Y':          list(tp['Y']),
            'Visibility': list(tp['Visibility']),
        }

        for start in range(0, n, seq_len):
            end       = min(start + seq_len, n)
            chunk_len = end - start

            coor = np.array(
                [[x_norm[i], y_norm[i]] for i in range(start, end)],
                dtype=np.float32,
            )
            msk = np.array(
                [[mask[i], mask[i]] for i in range(start, end)],
                dtype=np.float32,
            )

            # 補零至 seq_len
            if chunk_len < seq_len:
                pad = seq_len - chunk_len
                coor = np.pad(coor, ((0, pad), (0, 0)))
                msk  = np.pad(msk,  ((0, pad), (0, 0)))

            ct = torch.from_numpy(coor).unsqueeze(0).to(self.device)   # (1, L, 2)
            mt = torch.from_numpy(msk[:, :1]).unsqueeze(0).to(self.device)  # (1, L, 1)

            with torch.no_grad():
                out = self._inpaintnet(ct, mt).cpu()                    # (1, L, 2)

            ct_cpu = torch.from_numpy(coor).unsqueeze(0)
            mt_cpu = torch.from_numpy(msk[:, :1]).unsqueeze(0)
            out = out * mt_cpu + ct_cpu * (1 - mt_cpu)

            th = (out[0, :, 0] < COOR_TH) & (out[0, :, 1] < COOR_TH)
            out[0, th] = 0.0

            coor_np = out[0].numpy()
            for fi in range(chunk_len):
                i = start + fi
                if mask[i]:
                    px = int(coor_np[fi, 0] * orig_w)
                    py = int(coor_np[fi, 1] * orig_h)
                    result['X'][i]          = px
                    result['Y'][i]          = py
                    result['Visibility'][i] = 0 if px == 0 and py == 0 else 1

        return result
