"""
主視窗
介面佈局：
  ┌──────────────────────────────────────────────────────┐
  │  [選擇影片]  [開始分析]  [停止]  ████░░  進度/狀態   │
  ├─────────────────────────┬────────────────────────────┤
  │                         │  即時偵測紀錄               │
  │    影片畫面              │  [00:23] Smash ★★★☆☆ 62% │
  │    (含骨架疊加)          │  [00:31] Clear ★★★★☆ 81% │
  │                         │  ...                       │
  │                         ├────────────────────────────┤
  │                         │  整場報告（分析完後顯示）   │
  └─────────────────────────┴────────────────────────────┘
"""

import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from badminton.scoring.report_generator import generate_report, ms_to_timestamp, score_to_stars
from gui.analysis_worker import AnalysisWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 羽球動作分析系統")
        self.resize(1280, 720)

        self._worker = None
        self._video_path = ""

        self._build_ui()

    # ─────────────────────────────────────────
    # UI 建立
    # ─────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ── 頂部工具列 ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_open = QPushButton("選擇影片")
        self.btn_open.setFixedHeight(36)
        self.btn_open.clicked.connect(self._on_open)

        self.btn_start = QPushButton("開始分析")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setValue(0)

        self.lbl_status = QLabel("請選擇影片")
        self.lbl_status.setObjectName("lbl_status")
        self.lbl_status.setFixedHeight(36)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(self.progress_bar, stretch=1)
        toolbar.addWidget(self.lbl_status)
        root_layout.addLayout(toolbar)

        # ── 主內容區：左影片 + 右面板 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：影片畫面
        self.video_label = QLabel("尚未載入影片")
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        splitter.addWidget(self.video_label)

        # 右：結果面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        # ── 即時狀態卡片 ──
        lbl_stats = QLabel("即時狀態")
        lbl_stats.setObjectName("lbl_section")
        right_layout.addWidget(lbl_stats)

        stats_card = QFrame()
        stats_card.setObjectName("stats_card")
        card_layout = QVBoxLayout(stats_card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(6)

        # 手腕速度 row
        row_speed = QHBoxLayout()
        lbl_speed_key = QLabel("手腕速度")
        lbl_speed_key.setObjectName("stat_key")
        self.lbl_speed_val = QLabel("—")
        self.lbl_speed_val.setObjectName("stat_val")
        row_speed.addWidget(lbl_speed_key)
        row_speed.addStretch()
        row_speed.addWidget(self.lbl_speed_val)
        card_layout.addLayout(row_speed)

        # 球速 row
        row_ball = QHBoxLayout()
        lbl_ball_key = QLabel("球速（px/s）")
        lbl_ball_key.setObjectName("stat_key")
        self.lbl_ball_val = QLabel("—")
        self.lbl_ball_val.setObjectName("stat_val")
        row_ball.addWidget(lbl_ball_key)
        row_ball.addStretch()
        row_ball.addWidget(self.lbl_ball_val)
        card_layout.addLayout(row_ball)

        # 動作狀態 row
        row_ctx = QHBoxLayout()
        lbl_ctx_key = QLabel("動作狀態")
        lbl_ctx_key.setObjectName("stat_key")
        self.lbl_ctx_val = QLabel("—")
        self.lbl_ctx_val.setObjectName("stat_val")
        row_ctx.addWidget(lbl_ctx_key)
        row_ctx.addStretch()
        row_ctx.addWidget(self.lbl_ctx_val)
        card_layout.addLayout(row_ctx)

        card_sep = QFrame()
        card_sep.setFrameShape(QFrame.Shape.HLine)
        card_layout.addWidget(card_sep)

        # 動作計數 badges
        counts_row = QHBoxLayout()
        counts_row.setSpacing(6)
        self.count_labels = {}
        for name in ["殺球", "高遠球", "吊球", "平抽球", "切球"]:
            badge = QLabel(f"{name}\n0")
            badge.setObjectName("stat_count")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            counts_row.addWidget(badge)
            self.count_labels[name] = badge
        card_layout.addLayout(counts_row)

        right_layout.addWidget(stats_card)

        # ── 偵測紀錄 ──
        lbl_live = QLabel("偵測紀錄")
        lbl_live.setObjectName("lbl_section")
        right_layout.addWidget(lbl_live)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setFont(QFont("Courier New", 9))
        self.live_log.setFixedHeight(150)
        right_layout.addWidget(self.live_log)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        right_layout.addWidget(separator)

        lbl_report = QLabel("整場報告")
        lbl_report.setObjectName("lbl_section")
        right_layout.addWidget(lbl_report)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setFont(QFont("Courier New", 9))
        self.report_box.setPlaceholderText("分析完成後，完整報告會顯示在這裡。")
        right_layout.addWidget(self.report_box, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([820, 460])
        root_layout.addWidget(splitter, stretch=1)

    # ─────────────────────────────────────────
    # 按鈕事件
    # ─────────────────────────────────────────

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇羽球影片", "",
            "影片檔案 (*.mp4 *.avi *.mov *.mkv);;所有檔案 (*)"
        )
        if path:
            self._video_path = path
            self.lbl_status.setText(f"已選擇：{os.path.basename(path)}")
            self.btn_start.setEnabled(True)
            self.report_box.clear()
            self.live_log.clear()
            self.progress_bar.setValue(0)

    def _on_start(self):
        if not self._video_path:
            return

        self.live_log.clear()
        self.report_box.clear()
        self.progress_bar.setValue(0)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open.setEnabled(False)
        self.lbl_status.setText("分析中...")

        self.lbl_speed_val.setText("—")
        self.lbl_ball_val.setText("—")
        self.lbl_ctx_val.setText("—")
        for name, lbl in self.count_labels.items():
            lbl.setText(f"{name}\n0")

        self._worker = AnalysisWorker(self._video_path)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.action_found.connect(self._on_action)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.progress.connect(self._on_progress)
        self._worker.status_msg.connect(self.lbl_status.setText)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.lbl_status.setText("已停止")
        self._reset_buttons()

    # ─────────────────────────────────────────
    # Worker 信號處理
    # ─────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        """將 OpenCV 畫面轉成 Qt 可顯示的格式，更新到影片 Label。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    _CTX_ZH = {"offense": "進攻", "defense": "防守", "neutral": "待機"}

    def _on_stats(self, speed: float, context: str, counts: dict, ball_speed: float):
        """每幀更新即時狀態卡片。"""
        self.lbl_speed_val.setText(f"{speed:.2f}")
        self.lbl_ball_val.setText(f"{ball_speed:.0f}" if ball_speed > 0 else "—")
        self.lbl_ctx_val.setText(self._CTX_ZH.get(context, context))
        for name, lbl in self.count_labels.items():
            lbl.setText(f"{name}\n{counts.get(name, 0)}")

    def _on_action(self, action: str, dtw_score, advice: list):
        """每偵測到一個動作，在即時紀錄區新增一行。"""
        stars = score_to_stars(dtw_score)
        score_str = f"{dtw_score:.0f}%" if dtw_score is not None else "N/A"
        advice_str = advice[0] if advice else ""
        line = f"{action:<6} {stars}  {score_str:<6}  {advice_str}"
        self.live_log.append(line)

    def _on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)

    def _on_finished(self, event_log: list, total_ms: int):
        """分析完成，生成並顯示完整報告。"""
        self.progress_bar.setValue(100)
        self.lbl_status.setText("分析完成")
        self._reset_buttons()

        report = generate_report(
            event_log,
            video_name=os.path.basename(self._video_path),
            total_ms=total_ms,
        )
        self.report_box.setPlainText(report)

    def _on_error(self, msg: str):
        self.lbl_status.setText(f"錯誤：{msg}")
        self._reset_buttons()

    # ─────────────────────────────────────────
    # 工具
    # ─────────────────────────────────────────

    def _reset_buttons(self):
        self.btn_start.setEnabled(bool(self._video_path))
        self.btn_stop.setEnabled(False)
        self.btn_open.setEnabled(True)
