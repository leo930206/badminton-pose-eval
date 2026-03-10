# Badminton Pose Evaluation System
### 基於電腦視覺之羽球動作辨識與評估系統

**網頁工具：[www.xkuan.com/badminton](https://www.xkuan.com/badminton)**

---

## 專案簡介

透過電腦視覺自動分析羽球練習影片，辨識 12 種揮拍動作，並與職業選手模板比較給出評分與動作建議。提供桌面 GUI（本地端）與網頁工具（雲端）兩種使用方式。

---

## 功能

### 桌面 GUI（本地端）
- **影片載入**：選取本地影片，雙 Pass 分析（羽球追蹤 → 骨架 + 評分）
- **即時顯示**：骨架 33 點疊加 + 羽球軌跡殘影
- **12 種球種辨識**：殺球、高遠球、吊球、平抽球、切球、網前放小球、擋小球、挑球、推球、撲球、勾球、發短球 / 發長球
- **DTW 評分**：與 240 個職業選手模板比對，輸出星評（★★★ Excellent / Good / Fair）與動作建議
- **時間軸播放器**：拖曳 scrub 即時顯示骨架，80ms debounce 防卡頓，顯示 MM:SS 時間戳
- **逐球確認模式**：每次偵測到擊球自動暫停，顯示動作名稱、分數與建議，點擊畫面繼續
- **16:9 影片區域**：VideoLabel 強制維持比例

### 網頁工具（www.xkuan.com/badminton）
- 上傳影片 → 後端分析 → 顯示結果報告
- 4 色狀態燈（綠 / 黃 / 紅 / 灰）反映後端服務狀態
- 喚醒遮罩（Render 免費方案冷啟動 30 秒）
- 歷史記錄（localStorage，最多 100 筆，超過 80 筆提示匯出 CSV）
- Chart.js 進步曲線，可依球種篩選

---

## 技術架構

| 元件 | 技術 |
|------|------|
| 骨架追蹤 | MediaPipe Pose Landmarker（33 landmarks → 9 關節） |
| 羽球追蹤 | TrackNetV3（TrackNet + InpaintNet，CUDA 加速） |
| 動作偵測 | 規則式狀態機（直臂閾值 158°、冷卻 500ms） |
| 動作分類 | DTW 全球種比對（240 模板：12 種 × 20 個） |
| 模板來源 | CoachAI ShuttleSet（KDD 2023）職業選手骨架 |
| 桌面 GUI | PyQt6 |
| 網頁前端 | 純 HTML / CSS / JS（Cloudflare Pages） |
| 網頁後端 | FastAPI（Render，開發中） |
| 語言 | Python 3.10+ |

---

## 專案結構

```
badminton-pose-eval/
├── main.py                    # GUI 入口：python main.py
├── config.py                  # 全域參數
├── badminton/
│   ├── classification/        # detector.py（規則式偵測）
│   ├── scoring/               # dtw_scorer.py、report_generator.py
│   ├── data/                  # logger.py、sequence_buffer.py
│   └── display/               # renderer.py
├── gui/
│   ├── main_window.py         # PyQt6 主視窗
│   └── analysis_worker.py     # 分析執行緒（雙 Pass）
├── tools/
│   ├── build_dtw_templates.py # 建立 DTW 模板庫
│   └── extract_template.py
├── models/
│   └── pose_landmarker_lite.task
├── datasets/
│   ├── templates/             # DTW 模板（12 球種 × 20 個 JSON）
│   ├── shuttleset/            # CoachAI ShuttleSet（.gitignore）
│   └── raw/                   # 原始影片（.gitignore）
├── index.html                 # 網頁工具前端
└── output/                    # 分析輸出（.gitignore）
```

---

## 安裝步驟

### 1. Python 套件

```bash
pip install mediapipe opencv-python PyQt6 numpy scipy
```

PyTorch（有 NVIDIA GPU）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

PyTorch（CPU）：
```bash
pip install torch
```

### 2. TrackNetV3（羽球追蹤）

```bash
git clone https://github.com/alenzenx/TracknetV3.git tracknet
mkdir tracknet/ckpts
```

至 [TrackNetV3 Releases](https://github.com/alenzenx/TracknetV3) 下載預訓練模型，放到 `tracknet/ckpts/`：
- `TrackNet_best.pt`（約 130 MB）
- `InpaintNet_best.pt`（約 6 MB）

### 3. 建立 DTW 模板庫

需先下載 [CoachAI ShuttleSet](https://github.com/CoachAI/ShuttleSet) 放到 `datasets/shuttleset/`，然後：

```bash
python tools/build_dtw_templates.py --data datasets/shuttleset --samples 20
```

---

## 執行

```bash
python main.py
```

> **Windows + NVIDIA GPU**：`main.py` 確保 `torch` 在 PyQt6 之前載入，避免 CUDA DLL 衝突。

---

## 分析流程

```
Pass 1（0–50%）：TrackNetV3 偵測每幀羽球位置（CUDA 加速）
        ↓
Pass 2（50–100%）：MediaPipe 骨架分析
        ↓
        規則式偵測擊球時機（手肘角度 + 手腕速度）
        ↓
        DTW 比對 240 個模板 → 球種分類 + 評分
        ↓
        輸出結果（Badge 顯示 + 暫停 Overlay + 報告）
```

---

## 資料集

- **CoachAI ShuttleSet**（KDD 2023）：36,492 筆職業選手標注，12 種球種
- 資料位置：`datasets/shuttleset/`（.gitignore，需自行下載）
- 用途：建立 DTW 模板庫（hip-torso 正規化，9 關節，每球種取 20 個最佳片段）

---

## 部署（網頁工具）

- **Pages 網址**：`badminton-pose-eval.pages.dev`
- **自訂網域**：`www.xkuan.com/badminton`（Cloudflare Worker 路由）
- **後端**：FastAPI on Render（開發中）
