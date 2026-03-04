"""
GUI 主題樣式（macOS Light 簡約版）
Apple HIG 設計語言：
  - 背景：#f5f5f7（Apple 淺灰）
  - 卡片：#ffffff + 極細邊框
  - 強調色：#007aff（Apple Blue）
  - 文字層次：#1c1c1e / #6e6e73 / #aeaeb2
  - 圓角、細分隔線、無多餘背景色
"""

APP_STYLESHEET = """

/* ══════════════════════════════════════
   全域基礎
   ══════════════════════════════════════ */
QWidget {
    background-color: #f5f5f7;
    color: #1c1c1e;
    font-family: "Segoe UI Variable", "Segoe UI", "PingFang TC",
                 "Microsoft JhengHei UI", sans-serif;
    font-size: 14px;
}

/* ══════════════════════════════════════
   按鈕
   ══════════════════════════════════════ */
QPushButton {
    background-color: #ffffff;
    color: #1c1c1e;
    border: 1px solid #d1d1d6;
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 14px;
    font-weight: 500;
    min-width: 64px;
    min-height: 28px;
}
QPushButton:hover   { background-color: #f5f5f7; }
QPushButton:pressed { background-color: #e5e5ea; border-color: #c7c7cc; }
QPushButton:disabled {
    color: #c7c7cc;
    border-color: #e5e5ea;
    background-color: #fafafa;
}

/* 開始分析：主動作藍色按鈕 */
QPushButton#btn_start {
    background-color: #007aff;
    color: #ffffff;
    border: none;
    font-weight: 600;
}
QPushButton#btn_start:hover   { background-color: #0071eb; }
QPushButton#btn_start:pressed { background-color: #005ecb; }
QPushButton#btn_start:disabled {
    background-color: #cce0ff;
    color: #ffffff;
}

/* 停止：紅色警示按鈕 */
QPushButton#btn_stop:enabled {
    background-color: #ff3b30;
    color: #ffffff;
    border: none;
    font-weight: 600;
}
QPushButton#btn_stop:enabled:hover   { background-color: #e0342a; }
QPushButton#btn_stop:enabled:pressed { background-color: #c52d23; }

/* ══════════════════════════════════════
   CheckBox
   ══════════════════════════════════════ */
QCheckBox {
    color: #1c1c1e;
    font-size: 14px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1.5px solid #aeaeb2;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:hover   { border-color: #007aff; }
QCheckBox::indicator:checked {
    background-color: #007aff;
    border-color:     #007aff;
    image: url(none);
}

/* ══════════════════════════════════════
   進度條
   ══════════════════════════════════════ */
QProgressBar {
    background-color: #e5e5ea;
    border: none;
    border-radius: 3px;
    color: transparent;
    max-height: 6px;
}
QProgressBar::chunk {
    background-color: #007aff;
    border-radius: 3px;
}

/* ══════════════════════════════════════
   時間軸 Slider
   ══════════════════════════════════════ */
QSlider::groove:horizontal {
    height: 4px;
    background: #e5e5ea;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #007aff;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #007aff;
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}
QSlider::handle:horizontal:hover {
    background: #007aff;
}
QSlider::handle:horizontal:disabled {
    border-color: #c7c7cc;
    background: #f2f2f7;
}

/* ══════════════════════════════════════
   影片顯示區（無多餘虛線邊框）
   ══════════════════════════════════════ */
QLabel#video_label {
    background-color: #e8e8ed;
    border: none;
    border-radius: 12px;
    color: #8e8e93;
}

/* ══════════════════════════════════════
   時間標籤（MM:SS / MM:SS）
   ══════════════════════════════════════ */
QLabel#lbl_time {
    color: #aeaeb2;
    font-size: 13px;
}

/* ══════════════════════════════════════
   狀態文字
   ══════════════════════════════════════ */
QLabel#lbl_status {
    color: #6e6e73;
    font-size: 14px;
}

/* ══════════════════════════════════════
   區塊小標題（偵測紀錄 / 整場報告 …）
   ══════════════════════════════════════ */
QLabel#lbl_section {
    font-size: 11px;
    font-weight: 700;
    color: #aeaeb2;
    padding: 6px 0px 2px 0px;
}

/* ══════════════════════════════════════
   即時狀態卡片
   ══════════════════════════════════════ */
QFrame#stats_card {
    background-color: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 12px;
}

QLabel#stat_key {
    color: #8e8e93;
    font-size: 14px;
}
QLabel#stat_val {
    color: #1c1c1e;
    font-size: 14px;
    font-weight: 600;
}

/* 動作計數（殺球 / 高遠球 … 不需額外背景） */
QLabel#stat_count {
    background-color: transparent;
    border: none;
    padding: 2px 0px;
    font-size: 13px;
    color: #1c1c1e;
}

/* ══════════════════════════════════════
   文字框（即時紀錄 & 整場報告）
   ══════════════════════════════════════ */
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 8px;
    padding: 8px 10px;
    color: #1c1c1e;
    selection-background-color: #007aff;
    selection-color: #ffffff;
}

/* ══════════════════════════════════════
   分隔線
   ══════════════════════════════════════ */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    border: none;
    border-top: 1px solid #f0f0f5;
    max-height: 1px;
    background-color: transparent;
}

/* ══════════════════════════════════════
   Splitter 把手（幾乎隱形）
   ══════════════════════════════════════ */
QSplitter::handle:horizontal {
    background-color: #e5e5ea;
    width: 1px;
}

/* ══════════════════════════════════════
   捲動條（細條）
   ══════════════════════════════════════ */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #d1d1d6;
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover  { background: #aeaeb2; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical      { height: 0; }

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
}
QScrollBar::handle:horizontal {
    background: #d1d1d6;
    border-radius: 3px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #aeaeb2; }
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal     { width: 0; }

"""
