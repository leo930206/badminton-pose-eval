"""
主視窗
介面佈局：
  ┌──────────────────────────────────────────────────────────────┐
  │  [選擇影片]  [開始分析]  [停止]  [☑正常速度]  [☑擊球暫停]  進度/狀態 │
  ├──────────────────────────────┬───────────────────────────────┤
  │                              │  即時偵測紀錄                  │
  │    影片畫面（16:9）           │  [00:23] Smash ★★★☆☆ 62%    │
  │    (含骨架疊加)               │  ...                          │
  │    [= 擊球暫停遮罩 =]         ├───────────────────────────────┤
  │                              │  整場報告  [匯出報告]           │
  │  [======時間軸======]         │                               │
  │   MM:SS / MM:SS              │                               │
  └──────────────────────────────┴───────────────────────────────┘
"""

import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QSlider,
    QSizePolicy, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from badminton.scoring.report_generator import generate_report, ms_to_timestamp, score_to_stars
from gui.analysis_worker import AnalysisWorker


# ════════════════════════════════════════════════════
# 自訂元件
# ════════════════════════════════════════════════════

class _VideoLabel(QLabel):
    """強制 16:9 比例的影片顯示框。"""

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width * 9 // 16

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        return QSize(base.width(), self.heightForWidth(base.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for child in self.children():
            if not isinstance(child, QWidget):
                continue
            if isinstance(child, _HitBanner):
                bh = _HitBanner.BANNER_H
                child.setGeometry(0, self.height() - bh, self.width(), bh)
            else:
                child.setGeometry(self.rect())

    def mousePressEvent(self, event):
        """點擊影片區任何地方 → 若橫幅正在顯示則繼續分析。"""
        if event.button() == Qt.MouseButton.LeftButton:
            for child in self.children():
                if isinstance(child, _HitBanner) and child.isVisible():
                    child.hide()
                    child.resume_requested.emit()
                    return
        super().mousePressEvent(event)


class _HitBanner(QWidget):
    """底部橫幅式擊球提示（影片畫面不變暗，只在底部出現一條橫幅）。"""

    BANNER_H = 90           # 橫幅固定高度（px）
    resume_requested = pyqtSignal()

    _ACTION_MAP = {
        "殺球":  "🏸  Smash 殺球",
        "高遠球": "🏸  Clear 高遠球",
        "吊球":  "🏸  Drop 吊球",
        "平抽球": "🏸  Drive 平抽球",
        "切球":  "🏸  Cut 切球",
    }

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("hit_banner")
        self.setStyleSheet("""
            QWidget#hit_banner {
                background-color: rgba(0, 0, 0, 185);
                border-radius: 0px 0px 10px 10px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 9, 18, 10)
        layout.setSpacing(3)

        row1 = QHBoxLayout()
        self.lbl_action = QLabel()
        self.lbl_action.setStyleSheet(
            "color: white; font-size: 16px; font-weight: 700; background: transparent;"
        )
        self.lbl_score = QLabel()
        self.lbl_score.setStyleSheet(
            "color: #7ee787; font-size: 14px; background: transparent;"
        )
        row1.addWidget(self.lbl_action)
        row1.addStretch()
        row1.addWidget(self.lbl_score)

        self.lbl_advice = QLabel()
        self.lbl_advice.setWordWrap(True)
        self.lbl_advice.setStyleSheet(
            "color: #f8c555; font-size: 13px; background: transparent;"
        )

        self.lbl_hint = QLabel("▶  點擊此處繼續")
        self.lbl_hint.setStyleSheet(
            "color: rgba(255,255,255,160); font-size: 12px;"
            " font-style: italic; background: transparent;"
        )

        layout.addLayout(row1)
        layout.addWidget(self.lbl_advice)
        layout.addWidget(self.lbl_hint)

        self.hide()

    def show_hit(self, action: str, dtw_score, advice: list):
        self.lbl_action.setText(self._ACTION_MAP.get(action, action))
        self.lbl_score.setText(
            f"DTW {dtw_score:.0f}%" if dtw_score is not None else "DTW N/A"
        )
        self.lbl_advice.setText(
            "    ".join(f"• {a}" for a in advice[:2]) if advice else "（無具體建議）"
        )
        if self.parentWidget():
            p = self.parentWidget()
            self.setGeometry(0, p.height() - self.BANNER_H, p.width(), self.BANNER_H)
        self.raise_()
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.hide()
            self.resume_requested.emit()
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════
# 主視窗
# ════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("電腦視覺之羽球技術動作評估系統")
        self.resize(1280, 720)

        self._worker               = None
        self._video_path           = ""
        self._ball_positions: dict = {}
        self._frame_landmarks: dict = {}   # {frame_idx: [(norm_x, norm_y), ...]}
        self._fps: float           = 30.0
        self._pause_on_action      = False  # 本次分析是否啟用擊球暫停

        # 時間軸邊拖邊更新：週期計時器，拖動時每 50ms 更新一幀
        self._scrub_timer = QTimer()
        self._scrub_timer.setInterval(50)   # 50ms ≈ 20fps，非 single-shot
        self._scrub_timer.timeout.connect(self._do_scrub)
        self._pending_scrub_frame = -1
        self._scrub_cap: cv2.VideoCapture = None   # 持久化 cap，避免反覆開關檔案

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

        self.chk_speed = QCheckBox("正常速度")
        self.chk_speed.setToolTip("勾選後以影片原始幀率播放（較慢，適合仔細觀察）")

        self.chk_pause = QCheckBox("擊球暫停")
        self.chk_pause.setChecked(True)
        self.chk_pause.setToolTip(
            "分析時偵測到擊球動作，自動暫停並顯示建議分數（點影片任意處繼續）"
        )

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
        toolbar.addWidget(self.chk_speed)
        toolbar.addWidget(self.chk_pause)
        toolbar.addWidget(self.progress_bar, stretch=1)
        toolbar.addWidget(self.lbl_status)
        root_layout.addLayout(toolbar)

        # ── 主內容區：左影片 + 右面板 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：影片畫面 + 時間軸 slider
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # 16:9 強制比例影片標籤
        self.video_label = _VideoLabel("尚未載入影片")
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sp.setHeightForWidth(True)
        self.video_label.setSizePolicy(sp)
        left_layout.addWidget(self.video_label, stretch=1)

        # 擊球底部橫幅（疊加在影片底部，不加入 layout）
        self.hit_banner = _HitBanner(self.video_label)
        self.hit_banner.resume_requested.connect(self._on_resume)

        # 時間軸 slider
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.setValue(0)
        self.timeline.setEnabled(False)
        self.timeline.setFixedHeight(22)
        self.timeline.setToolTip("分析完成後可拖拉查看任意幀（含骨架 + 球軌跡）")
        self.timeline.sliderMoved.connect(self._on_timeline_scrub)
        self.timeline.sliderReleased.connect(self._on_timeline_released)
        left_layout.addWidget(self.timeline)

        # 時間顯示標籤（像 YouTube 那樣顯示 MM:SS / MM:SS）
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_time.setObjectName("lbl_time")
        self.lbl_time.setFixedHeight(16)
        self.lbl_time.setStyleSheet("font-size: 14px; color: #8c959f;")
        left_layout.addWidget(self.lbl_time)

        splitter.addWidget(left_widget)

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

        row_speed = QHBoxLayout()
        lbl_speed_key = QLabel("手腕速度")
        lbl_speed_key.setObjectName("stat_key")
        self.lbl_speed_val = QLabel("—")
        self.lbl_speed_val.setObjectName("stat_val")
        row_speed.addWidget(lbl_speed_key)
        row_speed.addStretch()
        row_speed.addWidget(self.lbl_speed_val)
        card_layout.addLayout(row_speed)

        row_ball = QHBoxLayout()
        lbl_ball_key = QLabel("球速（px/s）")
        lbl_ball_key.setObjectName("stat_key")
        self.lbl_ball_val = QLabel("—")
        self.lbl_ball_val.setObjectName("stat_val")
        row_ball.addWidget(lbl_ball_key)
        row_ball.addStretch()
        row_ball.addWidget(self.lbl_ball_val)
        card_layout.addLayout(row_ball)

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

        lbl_live = QLabel("偵測紀錄")
        lbl_live.setObjectName("lbl_section")
        right_layout.addWidget(lbl_live)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setFont(QFont("Courier New", 11))
        self.live_log.setFixedHeight(160)
        right_layout.addWidget(self.live_log)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        right_layout.addWidget(separator)

        report_header = QHBoxLayout()
        lbl_report = QLabel("整場報告")
        lbl_report.setObjectName("lbl_section")
        self.btn_export = QPushButton("匯出報告")
        self.btn_export.setFixedHeight(24)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_report)
        report_header.addWidget(lbl_report)
        report_header.addStretch()
        report_header.addWidget(self.btn_export)
        right_layout.addLayout(report_header)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setFont(QFont("Courier New", 11))
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
            filename = os.path.basename(path)
            self.lbl_status.setText(f"已選擇：{filename}")
            self.btn_start.setEnabled(True)
            self.report_box.clear()
            self.live_log.clear()
            self.progress_bar.setValue(0)
            self.timeline.setValue(0)
            self.timeline.setEnabled(False)
            self.lbl_time.setText("00:00 / 00:00")
            self._close_scrub_cap()
            # 影片區顯示已選擇的影片名稱
            self.video_label.clear()
            self.video_label.setText(f"📁  {filename}\n\n請按「開始分析」開始")

    def _on_start(self):
        if not self._video_path:
            return

        self.live_log.clear()
        self.report_box.clear()
        self.progress_bar.setValue(0)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open.setEnabled(False)
        self.chk_speed.setEnabled(False)
        self.chk_pause.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.lbl_status.setText("分析中...")

        self.timeline.setValue(0)
        self.timeline.setRange(0, 0)
        self.timeline.setEnabled(False)
        self.lbl_time.setText("00:00 / 00:00")

        self.lbl_speed_val.setText("—")
        self.lbl_ball_val.setText("—")
        self.lbl_ctx_val.setText("—")
        for name, lbl in self.count_labels.items():
            lbl.setText(f"{name}\n0")

        # Pass 1 期間影片區顯示提示（Pass 2 開始後會被影像覆蓋）
        self.video_label.clear()
        self.video_label.setText(
            "⏳  正在偵測羽球軌跡（Pass 1 / 2）…\n\n"
            "請稍候，分析完成後影片畫面將自動出現"
        )
        self._close_scrub_cap()

        self._pause_on_action = self.chk_pause.isChecked()

        self._worker = AnalysisWorker(
            self._video_path,
            normal_speed=self.chk_speed.isChecked(),
            pause_on_action=self._pause_on_action,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.action_found.connect(self._on_action)
        self._worker.action_found.connect(self._on_action_show_banner)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_progress.connect(self._on_frame_progress)
        self._worker.status_msg.connect(self.lbl_status.setText)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.hit_banner.hide()
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
        self.lbl_speed_val.setText(f"{speed:.2f}")
        self.lbl_ball_val.setText(f"{ball_speed:.0f}" if ball_speed > 0 else "—")
        self.lbl_ctx_val.setText(self._CTX_ZH.get(context, context))
        for name, lbl in self.count_labels.items():
            lbl.setText(f"{name}\n{counts.get(name, 0)}")

    def _on_action(self, action: str, dtw_score, advice: list, ts_ms: int):
        """每偵測到一個動作，在即時紀錄區新增一行。"""
        ts         = ms_to_timestamp(ts_ms)
        stars      = score_to_stars(dtw_score)
        score_str  = f"{dtw_score:.0f}%" if dtw_score is not None else "N/A"
        advice_str = advice[0] if advice else ""
        line = f"[{ts}] {action:<6} {stars}  {score_str:<6}  {advice_str}"
        self.live_log.append(line)

    def _on_action_show_banner(self, action: str, dtw_score, advice: list, ts_ms: int):
        """擊球瞬間：若已勾選「擊球暫停」，顯示底部橫幅等待使用者繼續。"""
        if self._pause_on_action:
            self.hit_banner.show_hit(action, dtw_score, advice)

    def _on_resume(self):
        """使用者點擊遮罩後，恢復分析執行緒。"""
        if self._worker:
            self._worker.resume()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _on_frame_progress(self, frame_idx: int, total_frames: int):
        """Pass 2 逐幀進度 → 更新時間軸與時間標籤。"""
        if total_frames > 0 and self.timeline.maximum() != total_frames - 1:
            self.timeline.setRange(0, total_frames - 1)
        self.timeline.setValue(frame_idx)
        self._update_time_label(frame_idx, total_frames)

    def _on_finished(
        self,
        event_log: list,
        total_ms: int,
        ball_positions: dict,
        total_frames: int,
        frame_landmarks: dict,
    ):
        """分析完成：生成報告、啟用時間軸與匯出按鈕。"""
        self.progress_bar.setValue(100)
        self.lbl_status.setText("分析完成")
        self._reset_buttons()

        self._ball_positions   = ball_positions
        self._frame_landmarks  = frame_landmarks
        self._fps = total_frames * 1000 / total_ms if total_ms > 0 else 30.0

        if total_frames > 0:
            self.timeline.setRange(0, total_frames - 1)
            self.timeline.setValue(total_frames - 1)
            self.timeline.setEnabled(True)
            self._update_time_label(total_frames - 1, total_frames)

        report = generate_report(
            event_log,
            video_name=os.path.basename(self._video_path),
            total_ms=total_ms,
        )
        self.report_box.setPlainText(report)
        self.btn_export.setEnabled(True)

    def _on_error(self, msg: str):
        self.lbl_status.setText(f"錯誤：{msg}")
        self._reset_buttons()

    # ─────────────────────────────────────────
    # 時間軸拖拉（節流 + 骨架 + 球軌跡疊加）
    # ─────────────────────────────────────────

    def _on_timeline_scrub(self, frame_idx: int):
        """用戶拖拉時間軸：立即更新時間標籤，周期計時器持續更新影像（邊拖邊看）。"""
        self._pending_scrub_frame = frame_idx
        self._update_time_label(frame_idx, self.timeline.maximum() + 1)
        if not self._scrub_timer.isActive():
            self._scrub_timer.start()   # 開始週期更新（50ms，非 single-shot）

    def _on_timeline_released(self):
        """放開拖拉：停止週期計時，確保最終幀精確顯示。"""
        self._scrub_timer.stop()
        self._do_scrub()

    def _update_time_label(self, frame_idx: int, total_frames: int):
        """更新時間軸下方的 MM:SS / MM:SS 顯示。"""
        if self._fps <= 0:
            return
        cur_sec   = frame_idx / self._fps
        total_sec = max(total_frames, 1) / self._fps
        self.lbl_time.setText(
            f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
            f"  /  "
            f"{int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
        )

    def _do_scrub(self):
        """週期計時器觸發：讀取對應幀並疊加骨架 + 球軌跡後顯示。"""
        frame_idx = self._pending_scrub_frame
        if frame_idx < 0 or not self._video_path:
            return

        # 使用持久化 cap 避免每次重新開啟檔案（加速連續拖拉）
        if self._scrub_cap is None or not self._scrub_cap.isOpened():
            self._scrub_cap = cv2.VideoCapture(self._video_path)

        self._scrub_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._scrub_cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]

        # ── 骨架疊加（使用分析時儲存的歸一化座標）──
        lm_data = self._frame_landmarks.get(frame_idx)
        if lm_data:
            try:
                from mediapipe.tasks.python.vision import pose_landmarker as _mp_pose
                connections = _mp_pose.PoseLandmarksConnections.POSE_LANDMARKS
                color = (0, 255, 0)
                for conn in connections:
                    if conn.start < len(lm_data) and conn.end < len(lm_data):
                        x1 = int(lm_data[conn.start][0] * w)
                        y1 = int(lm_data[conn.start][1] * h)
                        x2 = int(lm_data[conn.end][0] * w)
                        y2 = int(lm_data[conn.end][1] * h)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                for lm_x, lm_y in lm_data:
                    cv2.circle(frame, (int(lm_x * w), int(lm_y * h)), 3, color, -1)
            except Exception:
                pass

        # ── 球軌跡殘影（往前 12 幀）──
        trail_len = 12
        for i in range(trail_len):
            idx = frame_idx - i
            if idx < 0:
                break
            pos = self._ball_positions.get(idx)
            if pos:
                alpha  = 1.0 - i / trail_len
                radius = max(2, int(6 * alpha))
                color  = (0, int(140 * alpha), int(255 * alpha))
                x, y   = pos
                cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), radius,     color,      -1)

        # ── 時間戳（白字黑邊）──
        ts_sec  = frame_idx / self._fps
        ts_text = f"{int(ts_sec // 60):02d}:{int(ts_sec % 60):02d}"
        cv2.putText(frame, ts_text, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, ts_text, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        self._on_frame(frame)

    # ─────────────────────────────────────────
    # 匯出報告
    # ─────────────────────────────────────────

    def _on_export_report(self):
        text = self.report_box.toPlainText()
        if not text:
            return
        default_name = os.path.splitext(os.path.basename(self._video_path))[0] + "_report.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出報告", default_name,
            "文字檔案 (*.txt);;所有檔案 (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.lbl_status.setText(f"報告已匯出：{os.path.basename(path)}")

    # ─────────────────────────────────────────
    # 工具
    # ─────────────────────────────────────────

    def _reset_buttons(self):
        self.btn_start.setEnabled(bool(self._video_path))
        self.btn_stop.setEnabled(False)
        self.btn_open.setEnabled(True)
        self.chk_speed.setEnabled(True)
        self.chk_pause.setEnabled(True)

    def _close_scrub_cap(self):
        """釋放持久化 scrub VideoCapture（切換影片或關閉視窗時呼叫）。"""
        if self._scrub_cap is not None:
            self._scrub_cap.release()
            self._scrub_cap = None

    def closeEvent(self, event):
        self._close_scrub_cap()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        super().closeEvent(event)
