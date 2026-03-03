# Badminton Pose Evaluation System
### 基於電腦視覺之羽球動作辨識與評估系統

> 成功大學 NCKU｜1142 不分系整合專題｜官裕宸

---

## 專案簡介

本系統使用電腦視覺技術，透過攝影機或影片自動辨識羽球選手的揮拍動作，並根據動作品質給予即時評分與建議回饋。

## 功能

- **動作辨識**：自動偵測 5 種主要揮拍動作
  - Smash（殺球）、Clear（高遠球）、Drop（放網）、Drive（平抽）、Cut（切球）
- **即時評分**：依據關節角度、揮拍速度等指標給出 Excellent / Good / Fair 評級
- **動作比對**：透過 DTW（動態時間規整）與職業選手模板比較相似度
- **圖形介面**：PyQt5 GUI 可載入影片、即時顯示分析結果
- **報告輸出**：整場分析完畢後輸出 JSON 格式動作記錄

## 技術架構

| 元件 | 技術 |
|------|------|
| 骨架追蹤 | MediaPipe Pose Landmarker |
| 動作辨識 | 規則式狀態機 |
| 評分比對 | DTW（Dynamic Time Warping） |
| 圖形介面 | PyQt5 |
| 語言 | Python 3.10+ |

## 專案結構

```
AI_Badminton_Test/
├── main.py              # 命令列執行入口
├── gui_main.py          # GUI 執行入口
├── config.py            # 所有參數設定
├── badminton/           # 核心模組
│   ├── pose/            # 骨架追蹤
│   ├── classification/  # 動作辨識狀態機
│   ├── scoring/         # 評分與報告
│   ├── data/            # 資料記錄
│   └── display/         # 畫面渲染
├── gui/                 # GUI 元件
├── tools/               # 工具腳本（模板擷取等）
├── models/              # AI 模型檔案
├── datasets/
│   ├── templates/       # 職業選手 DTW 模板
│   └── raw/             # 原始教學影片（不上傳）
├── docs/                # 文件資料
└── output/              # 分析輸出（不上傳）
```

## 環境需求

```
Python 3.10+
mediapipe
opencv-python
PyQt5
numpy
```

安裝依賴：

```bash
pip install mediapipe opencv-python PyQt5 numpy
```

## 執行方式

**命令列模式：**
```bash
python main.py
```

**GUI 模式：**
```bash
python gui_main.py
```

測試影片預設路徑為 `datasets/test.mp4`，可在 `config.py` 修改。

---

## 進度追蹤

本專案附有網頁版進度管理工具，部署於 GitHub Pages：
[https://leo930206.github.io/badminton-pose-eval/](https://leo930206.github.io/badminton-pose-eval/)
