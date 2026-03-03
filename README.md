# Badminton Pose Evaluation System
### 電腦視覺之羽球技術動作評估系統

---

## 專案簡介

本系統使用電腦視覺技術，透過影片自動辨識羽球選手的揮拍動作，並根據動作品質給予即時評分與建議回饋。

## 功能

- **羽球追蹤**：整合 TrackNetV3 偵測羽球軌跡，在畫面上繪製殘影
- **動作辨識**：自動偵測 5 種主要揮拍動作
  - 殺球 / 高遠球 / 吊球 / 平抽球 / 切球
- **即時評分**：依據關節角度、揮拍速度等指標給出 Excellent / Good / Fair 評級
- **動作比對**：透過 DTW（動態時間規整）與職業選手模板比較相似度
- **即時統計**：腕部速度、球速（px/s）、各動作次數計數
- **圖形介面**：PyQt5 GUI 載入影片、即時顯示骨架 + 軌跡
- **報告輸出**：整場分析後輸出含球速、擊球高度的動作記錄（JSON）

## 技術架構

| 元件 | 技術 |
|------|------|
| 骨架追蹤 | MediaPipe Pose Landmarker |
| 羽球追蹤 | TrackNetV3（TrackNet + InpaintNet） |
| 動作辨識 | 規則式狀態機 |
| 評分比對 | DTW（Dynamic Time Warping） |
| 圖形介面 | PyQt5 |
| 語言 | Python 3.10+ |
| GPU 加速 | CUDA（選用，有 NVIDIA GPU 時自動啟用） |

## 專案結構

```
badminton-pose-eval/
├── main.py              # 執行入口：python main.py
├── config.py            # 所有參數設定
├── badminton/           # 核心模組
│   ├── pose/            # 骨架追蹤（model_loader, tracker）
│   ├── classification/  # 動作辨識狀態機（context, detector）
│   ├── scoring/         # 評分與報告（rule_engine, dtw_scorer, report_generator）
│   ├── data/            # 資料記錄（logger, sequence_buffer）
│   ├── display/         # 畫面渲染（renderer）
│   └── tracking/        # 羽球追蹤（shuttle_tracker）
├── gui/                 # GUI 元件（main_window, analysis_worker）
├── tools/               # 工具腳本（模板擷取等）
├── models/              # MediaPipe 模型（pose_landmarker_lite.task）
├── datasets/
│   ├── templates/       # 職業選手 DTW 模板（建立後使用）
│   └── raw/             # 原始教學影片（不上傳 git）
├── docs/                # 專題文件
├── tracknet/            # TrackNetV3（需手動安裝，見下方說明，不上傳 git）
└── output/              # 分析輸出（不上傳 git）
```

## 安裝步驟

### 1. 安裝 Python 套件

```bash
pip install mediapipe opencv-python PyQt5 numpy scipy
```

PyTorch（CUDA 版，需有 NVIDIA GPU）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

PyTorch（CPU 版，無 GPU 時）：
```bash
pip install torch
```

### 2. 安裝 TrackNetV3（羽球追蹤）

```bash
# clone TrackNetV3 原始碼到 tracknet/ 資料夾
git clone https://github.com/alenzenx/TracknetV3.git tracknet

# 建立 ckpts 目錄
mkdir tracknet/ckpts
```

接著至 [TrackNetV3 Releases](https://github.com/alenzenx/TracknetV3) 下載預訓練模型，放到 `tracknet/ckpts/`：
- `TrackNet_best.pt`（約 130 MB）
- `InpaintNet_best.pt`（約 6 MB）

### 3. MediaPipe 模型

首次執行時程式會自動下載，或手動放置：
```
models/pose_landmarker_lite.task
```

## 執行方式

```bash
python main.py
```

> **Windows + NVIDIA GPU 注意**：`main.py` 已確保 `torch` 在 PyQt5 之前載入，避免 CUDA DLL 衝突。

---

## 分析流程

分析分兩趟進行：

1. **Pass 1（0–50%）**：TrackNetV3 偵測每幀羽球位置（CUDA 加速）
2. **Pass 2（50–100%）**：MediaPipe 骨架分析 + 動作辨識 + DTW 評分

---

