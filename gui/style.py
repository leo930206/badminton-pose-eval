"""
GUI 主題樣式（淺色簡約版）
靈感來自 macOS / Apple HIG：
  - 背景：#f5f5f7（淺灰）
  - 白色卡片區：按鈕、文字框
  - 強調色：#007aff（Apple Blue）
  - 警示色：#ff3b30（Apple Red）
"""

APP_STYLESHEET = """

/* ── 全域基礎 ── */
QWidget {
    background-color: #f2f2f7;
    color: #1c1c1e;
    font-family: "Segoe UI Variable", "Segoe UI", "PingFang TC", "Microsoft JhengHei UI", sans-serif;
    font-size: 16px;
}

/* ── 普通按鈕（白底灰框） ── */
QPushButton {
    background-color: #ffffff;
    color: #1c1c1e;
    border: 1px solid #c6c6c8;
    border-radius: 7px;
    padding: 5px 16px;
    font-weight: 500;
    min-width: 72px;
}
QPushButton:hover   { background-color: #f2f2f7; }
QPushButton:pressed { background-color: #e5e5ea; }
QPushButton:disabled {
    color: #adadb8;
    border-color: #e5e5ea;
    background-color: #f9f9f9;
}

/* ── 開始分析：藍色主按鈕 ── */
QPushButton#btn_start {
    background-color: #007aff;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#btn_start:hover   { background-color: #0071eb; }
QPushButton#btn_start:pressed { background-color: #0062cc; }
QPushButton#btn_start:disabled {
    background-color: #b8d4ff;
    color: #ffffff;
}

/* ── 停止：紅色警示按鈕 ── */
QPushButton#btn_stop:enabled {
    background-color: #ff3b30;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#btn_stop:enabled:hover   { background-color: #e0342a; }
QPushButton#btn_stop:enabled:pressed { background-color: #c52d23; }

/* ── 進度條 ── */
QProgressBar {
    background-color: #e5e5ea;
    border: none;
    border-radius: 3px;
    color: transparent;
}
QProgressBar::chunk {
    background-color: #007aff;
    border-radius: 3px;
}

/* ── 影片顯示區（空白時淺色佔位符；有影像時被 pixmap 覆蓋，背景不可見） ── */
QLabel#video_label {
    background-color: #e5e5ea;
    border: 2px dashed #c7c7cc;
    border-radius: 10px;
    color: #8e8e93;
}

/* ── 區塊小標題（LIVE LOG / 整場報告） ── */
QLabel#lbl_section {
    font-size: 14px;
    font-weight: 700;
    color: #6e6e73;
    padding: 6px 0px 2px 0px;
}

/* ── 狀態文字 ── */
QLabel#lbl_status {
    color: #6e6e73;
    font-size: 15px;
}

/* ── 文字框（即時紀錄 & 整場報告） ── */
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #c6c6c8;
    border-radius: 8px;
    padding: 8px 10px;
    color: #1c1c1e;
    selection-background-color: #007aff;
    selection-color: white;
}

/* ── 分隔線 ── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    border: none;
    border-top: 1px solid #e5e5ea;
    max-height: 1px;
    background-color: transparent;
}

/* ── Splitter 分隔把手 ── */
QSplitter::handle:horizontal {
    background-color: #e5e5ea;
    width: 1px;
}

/* ── 即時狀態卡片外框 ── */
QFrame#stats_card {
    background-color: #ffffff;
    border: 1px solid #c6c6c8;
    border-radius: 10px;
}

/* ── 卡片內 Key（左側灰色說明文字） ── */
QLabel#stat_key {
    color: #6e6e73;
    font-size: 15px;
}

/* ── 卡片內 Value（右側數值，粗體） ── */
QLabel#stat_val {
    color: #1d1d1f;
    font-size: 15px;
    font-weight: 600;
}

/* ── 動作計數 Badge ── */
QLabel#stat_count {
    background-color: #f5f5f7;
    border: 1px solid #e5e5ea;
    border-radius: 6px;
    padding: 4px 2px;
    font-size: 14px;
    color: #1d1d1f;
    min-width: 52px;
}

/* ── 捲動條（細條風格） ── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #c7c7cc;
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #aeaeb2; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
}
QScrollBar::handle:horizontal {
    background: #c7c7cc;
    border-radius: 3px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #aeaeb2; }
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal { width: 0; }

"""
