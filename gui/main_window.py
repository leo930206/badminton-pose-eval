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
import webbrowser

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QSlider,
    QSizePolicy, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from badminton.scoring.report_generator import (
    generate_html_report, ms_to_timestamp,
    _grade_label, _grade_color, _ACTION_COLOR,
    _wrap_html_page, _wrap_html_page_qt, _hit_height_label,
)
from config import BALL_SPEED_KMH_SCALE
from gui.analysis_worker import AnalysisWorker

# 各動作計數 Badge 的淡色背景（動作色 15% 混白）
_BADGE_BG = {
    # ── 規則式 6 種 ──────────────────────────────────
    "殺球":  "#ffe0e6",   # 緋紅淡粉
    "高遠球": "#f3e5fa",  # 紫羅蘭淡紫
    "吊球":  "#ffedd9",   # 橘橙淡橘
    "平抽球": "#fff8d9",  # 金黃淡黃
    "切球":  "#e1f7e6",   # 草綠淡綠
    "挑球":  "#d9f6f5",   # 薄荷藍綠淡色
    # ── ShuttleSet 12 種（ML 分類器）────────────────
    "放小球": "#fff3e0",  # 橘黃淡
    "擋小球": "#e0f7ea",  # 青綠淡
    "長球":  "#f3e5fa",  # 紫淡
    "平球":  "#fffde0",  # 黃淡
    "推球":  "#e0f4ff",  # 天藍淡
    "撲球":  "#ffe0ea",  # 桃紅淡
    "勾球":  "#f5e0ff",  # 淺紫淡
    "發短球": "#e0f0ff",  # 藍淡
    "發長球": "#d9e8ff",  # 深藍淡
}


def _make_shadow(blur: int = 12, dy: int = 2, opacity: int = 22) -> QGraphicsDropShadowEffect:
    """建立 macOS 風格卡片投影。每個 widget 需獨立建立實例。"""
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(blur)
    fx.setOffset(0, dy)
    fx.setColor(QColor(0, 0, 0, opacity))
    return fx


# ════════════════════════════════════════════════════
# 自訂元件
# ════════════════════════════════════════════════════

class _VideoLabel(QLabel):
    """強制 16:9 比例的影片顯示框。"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._disp_pixmap: QPixmap | None = None

    def setDisplayPixmap(self, pixmap: QPixmap) -> None:
        """設定影片幀。使用 update() 觸發重繪，不呼叫 setPixmap()。
        setPixmap() 內部會呼叫 updateGeometry()，通知 layout 重算，
        導致視窗隨播放不斷放大。改用 paintEvent 自行繪製可完全避免此問題。"""
        self._disp_pixmap = pixmap
        self.update()   # 只重繪，不動 layout

    def clear(self):
        self._disp_pixmap = None
        super().clear()

    def setText(self, text: str):
        self._disp_pixmap = None
        super().setText(text)

    def paintEvent(self, event):
        if self._disp_pixmap is not None and not self._disp_pixmap.isNull():
            painter = QPainter(self)
            scaled = self._disp_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            super().paintEvent(event)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width * 9 // 16

    def sizeHint(self) -> QSize:
        return QSize(160, 90)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

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
    """底部橫幅式擊球提示（動作顏色左側強調小卡，疊加於影片底部）。"""

    BANNER_H = 88
    resume_requested = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("hit_banner")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 10)
        layout.setSpacing(3)

        # 第一行：動作名稱（左）+ 分數/等級（右）
        row1 = QHBoxLayout()
        self.lbl_action = QLabel()
        self.lbl_score  = QLabel()
        row1.addWidget(self.lbl_action)
        row1.addStretch()
        row1.addWidget(self.lbl_score)

        # 第二行：建議文字
        self.lbl_advice = QLabel()
        self.lbl_advice.setWordWrap(True)
        self.lbl_advice.setStyleSheet("color: #3c3c43; font-size: 12px; background: transparent;")

        # 第三行：繼續提示
        self.lbl_hint = QLabel("點擊任意處繼續")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_hint.setStyleSheet("color: #aeaeb2; font-size: 11px; background: transparent;")

        layout.addLayout(row1)
        layout.addWidget(self.lbl_advice)
        layout.addWidget(self.lbl_hint)

        self.hide()

    def show_hit(self, action: str, dtw_score, advice: list):
        action_color = _ACTION_COLOR.get(action, "#00c7be")
        grade        = _grade_label(dtw_score)
        grade_color  = _grade_color(dtw_score)

        # 動態設定左側強調色
        self.setStyleSheet(f"""
            QWidget#hit_banner {{
                background-color: rgba(250, 250, 252, 245);
                border-left: 5px solid {action_color};
                border-top: 1px solid rgba(0, 0, 0, 18);
                border-radius: 0px 0px 12px 12px;
            }}
        """)

        self.lbl_action.setText(action)
        self.lbl_action.setStyleSheet(
            "color: #007aff; font-size: 17px; font-weight: 700; background: transparent;"
        )

        score_txt = f"{dtw_score:.0f}%" if dtw_score is not None else "N/A"
        self.lbl_score.setText(f"{score_txt}  {grade}")
        self.lbl_score.setStyleSheet(
            f"color: {grade_color}; font-size: 13px; font-weight: 600; background: transparent;"
        )

        self.lbl_advice.setText(
            "  ".join(f"• {a}" for a in advice[:2]) if advice else "（無具體建議）"
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
        self._event_log: list      = []    # 最後一次分析結果（供匯出使用）
        self._total_ms: int        = 0

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
        root_layout.setContentsMargins(12, 10, 12, 12)
        root_layout.setSpacing(8)

        # ── 頂部工具列（白色圓角卡片）──
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("toolbar_frame")
        toolbar = QHBoxLayout(toolbar_frame)   # 直接用 HBox，不需外層 VBox
        toolbar.setContentsMargins(12, 6, 12, 6)
        toolbar.setSpacing(8)

        self.btn_open = QPushButton("選擇影片")
        self.btn_open.setFixedHeight(34)
        self.btn_open.clicked.connect(self._on_open)

        self.btn_start = QPushButton("開始分析")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(34)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        # 按鈕 / Checkbox 分隔線
        vline = QFrame()
        vline.setFixedWidth(1)
        vline.setFixedHeight(22)
        vline.setStyleSheet("background-color: #d1d1d6; border: none;")

        self.chk_speed = QCheckBox("正常速度")
        self.chk_speed.setToolTip("勾選後以影片原始幀率播放（較慢，適合仔細觀察）")

        self.chk_pause = QCheckBox("擊球暫停")
        self.chk_pause.setChecked(True)
        self.chk_pause.setToolTip(
            "分析時偵測到擊球動作，自動暫停並顯示建議分數（點影片任意處繼續）"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)

        self.lbl_status = QLabel("請選擇影片")
        self.lbl_status.setObjectName("lbl_status")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(vline, alignment=Qt.AlignmentFlag.AlignVCenter)
        toolbar.addWidget(self.chk_speed)
        toolbar.addWidget(self.chk_pause)
        toolbar.addWidget(self.progress_bar, stretch=1)   # 進度條在 擊球暫停 右邊
        toolbar.addWidget(self.lbl_status)

        toolbar_frame.setGraphicsEffect(_make_shadow(blur=14, dy=2, opacity=18))
        root_layout.addWidget(toolbar_frame)

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
        self.lbl_time.setFixedHeight(18)
        left_layout.addWidget(self.lbl_time)

        splitter.addWidget(left_widget)

        # 右：結果面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 4, 6, 4)
        right_layout.setSpacing(4)

        # ── 即時狀態卡片 ──
        lbl_stats = QLabel("即時狀態")
        lbl_stats.setObjectName("lbl_section")
        right_layout.addWidget(lbl_stats)

        stats_card = QFrame()
        stats_card.setObjectName("stats_card")
        card_layout = QVBoxLayout(stats_card)
        card_layout.setContentsMargins(12, 12, 12, 10)
        card_layout.setSpacing(8)

        # 三欄式：每欄「標題（上）＋ 數值（下）」垂直排列
        stats_row = QHBoxLayout()
        stats_row.setSpacing(0)
        for i, (key_text, attr_name) in enumerate([
            ("手腕速度 n/s", "lbl_speed_val"),
            ("球速 km/h", "lbl_ball_val"),
            ("動作狀態", "lbl_ctx_val"),
        ]):
            if i > 0:
                div = QFrame()
                div.setFixedWidth(1)
                div.setStyleSheet("background-color: #e5e5ea; border: none;")
                stats_row.addWidget(div)
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)
            key_lbl = QLabel(key_text)
            key_lbl.setObjectName("stat_key")
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl = QLabel("—")
            val_lbl.setObjectName("stat_val")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            setattr(self, attr_name, val_lbl)
            col.addWidget(key_lbl)
            col.addWidget(val_lbl)
            stats_row.addLayout(col, stretch=1)
        card_layout.addLayout(stats_row)

        card_sep = QFrame()
        card_sep.setFrameShape(QFrame.Shape.HLine)
        card_layout.addWidget(card_sep)

        # 兩列 badge：每列 6 種球種
        _BADGE_NAMES_ROW1 = ["殺球", "挑球", "長球", "放小球", "切球", "平球"]
        _BADGE_NAMES_ROW2 = ["擋小球", "推球", "撲球", "勾球", "發短球", "發長球"]
        self.count_labels = {}
        for row_names in [_BADGE_NAMES_ROW1, _BADGE_NAMES_ROW2]:
            counts_row = QHBoxLayout()
            counts_row.setSpacing(4)
            for name in row_names:
                badge = QLabel(f"{name}\n0")
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                bg = _BADGE_BG.get(name, "#f2f2f7")
                badge.setStyleSheet(
                    f"background-color: {bg}; border-radius: 6px; padding: 3px 4px;"
                    f" font-size: 11px; font-weight: 600; color: #1c1c1e; min-width: 38px;"
                )
                counts_row.addWidget(badge)
                self.count_labels[name] = badge
            card_layout.addLayout(counts_row)

        stats_card.setGraphicsEffect(_make_shadow())
        right_layout.addWidget(stats_card)

        lbl_live = QLabel("偵測紀錄")
        lbl_live.setObjectName("lbl_section")
        right_layout.addWidget(lbl_live)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setFixedHeight(168)            # 比例字型稍高
        self.live_log.setGraphicsEffect(_make_shadow(blur=10, dy=1, opacity=18))
        right_layout.addWidget(self.live_log)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        right_layout.addWidget(separator)

        report_header = QHBoxLayout()
        report_header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_report = QLabel("整場報告")
        lbl_report.setObjectName("lbl_section")
        self.btn_export = QPushButton("匯出報告")
        self.btn_export.setObjectName("btn_export")
        self.btn_export.setFixedHeight(22)
        self.btn_export.setFixedWidth(60)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_report)
        report_header.addWidget(lbl_report, 0, Qt.AlignmentFlag.AlignVCenter)
        report_header.addStretch()
        report_header.addWidget(self.btn_export, 0, Qt.AlignmentFlag.AlignVCenter)
        right_layout.addLayout(report_header)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setPlaceholderText("分析完成後，完整報告會顯示在這裡。")
        self.report_box.setGraphicsEffect(_make_shadow(blur=10, dy=1, opacity=18))
        right_layout.addWidget(self.report_box, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([800, 460])
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
        self.video_label.setDisplayPixmap(QPixmap.fromImage(q_img))

    _CTX_ZH = {"offense": "進攻", "defense": "防守", "neutral": "待機"}

    def _on_stats(self, speed: float, context: str, counts: dict, ball_speed: float):
        self.lbl_speed_val.setText(f"{speed:.2f}")
        self.lbl_ball_val.setText(f"{ball_speed * BALL_SPEED_KMH_SCALE:.0f}" if ball_speed > 0 else "—")
        self.lbl_ctx_val.setText(self._CTX_ZH.get(context, context))
        for name, lbl in self.count_labels.items():
            lbl.setText(f"{name}\n{counts.get(name, 0)}")

    def _on_action(self, action: str, dtw_score, advice: list, ts_ms: int,
                   ball_speed: float = 0.0, hit_height: float = 0.0):
        """每偵測到一個動作，以圓點格式新增至即時紀錄區（含球速與高度）。"""
        ts           = ms_to_timestamp(ts_ms)
        grade        = _grade_label(dtw_score)
        grade_color  = _grade_color(dtw_score)
        action_color = _ACTION_COLOR.get(action, "#00c7be")
        advice_str   = advice[0] if advice else "—"

        extra_parts = []
        if ball_speed > 0:   extra_parts.append(f"球速 {ball_speed * BALL_SPEED_KMH_SCALE:.0f}km/h")
        if hit_height > 0:   extra_parts.append(_hit_height_label(hit_height))
        extra_str = "  ".join(extra_parts)

        html = (
            f'<p style="margin:0 0 6px 0; padding:2px 6px; line-height:1.6;">'
            f'<span style="color:{action_color}; font-size:15px;">●</span>&nbsp;'
            f'<b style="color:#007aff; font-size:15px;">{action}</b>'
            f'&nbsp;&nbsp;<span style="color:{grade_color}; font-size:12px;'
            f' font-weight:600;">{grade}</span>'
            f'&nbsp;&nbsp;<span style="color:#aeaeb2; font-size:11px;">{ts}</span>'
            f'<br/>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;'
            f'<span style="color:#6e6e73; font-size:12px;">{advice_str}</span>'
            + (f'&nbsp;&nbsp;<span style="color:#aeaeb2; font-size:11px;">{extra_str}</span>'
               if extra_str else '')
            + f'</p>'
        )
        self.live_log.append(html)

    def _on_action_show_banner(self, action: str, dtw_score, advice: list, ts_ms: int,
                               ball_speed: float = 0.0, hit_height: float = 0.0):
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
        self._event_log        = event_log
        self._total_ms         = total_ms

        if total_frames > 0:
            self.timeline.setRange(0, total_frames - 1)
            self.timeline.setValue(total_frames - 1)
            self.timeline.setEnabled(True)
            self._update_time_label(total_frames - 1, total_frames)

        body_html = generate_html_report(
            event_log,
            video_name=os.path.basename(self._video_path),
            total_ms=total_ms,
        )
        self.report_box.setHtml(_wrap_html_page_qt(body_html))
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
        if not self._event_log:
            return
        video_name   = os.path.basename(self._video_path)
        default_name = os.path.splitext(video_name)[0] + " 評估報告.html"
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出 HTML 報告", default_name,
            "HTML 報告 (*.html);;所有檔案 (*)"
        )
        if not path:
            return
        body_html = generate_html_report(
            self._event_log,
            video_name=video_name,
            total_ms=self._total_ms,
            include_shot_log=True,
        )
        full_html = _wrap_html_page(body_html, video_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_html)
        self.lbl_status.setText(f"報告已匯出：{os.path.basename(path)}")
        webbrowser.open(f"file:///{path.replace(os.sep, '/')}")

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

    def keyPressEvent(self, event):
        """空白鍵：等同點擊畫面繼續分析（擊球暫停中有效）。"""
        if event.key() == Qt.Key.Key_Space and self.hit_banner.isVisible():
            self.hit_banner.hide()
            self._on_resume()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._close_scrub_cap()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        super().closeEvent(event)
