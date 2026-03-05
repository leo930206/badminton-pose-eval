"""
整場分析報告產生器
用途：把分析過程中累積的 event_log 轉成 HTML 報告。
"""

from config import BALL_SPEED_KMH_SCALE


def ms_to_timestamp(ms: int) -> str:
    """將毫秒轉成 MM:SS 格式。"""
    total_sec = ms // 1000
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"



# ── 動作顏色表（ShuttleSet 12 種球種）────────────────────────────────────────
# 若需修改顏色，在此改即可（同時影響 HTML 報告 badge 和 GUI 擊球橫幅色條）
# main_window.py 的 _BADGE_BG 是統計區 badge 背景色，也需一起修改
_ACTION_COLOR = {
    "殺球":  "#ff2d55",   # 緋紅
    "挑球":  "#00c7be",   # 薄荷藍綠
    "長球":  "#af52de",   # 紫
    "放小球": "#ff9f0a",  # 橘黃
    "切球":  "#34c759",   # 草綠
    "平球":  "#ffd60a",   # 黃
    "擋小球": "#30d158",  # 青綠
    "推球":  "#64d2ff",   # 天藍
    "撲球":  "#ff375f",   # 桃紅
    "勾球":  "#bf5af2",   # 淺紫
    "發短球": "#32ade6",  # 藍
    "發長球": "#0a84ff",  # 深藍
}
# 備用色（規則式退路：高遠球/吊球/平抽球 偶爾出現時使用）
_BACKUP_COLORS = ["#636366", "#48484a"]


def _hit_height_label(h: float) -> str:
    """將擊球高度比例（0=底部、1=頂部）轉成直覺文字。"""
    if h >= 0.65:
        return "高球位"
    if h >= 0.35:
        return "中球位"
    return "低球位"


def _grade_label(score) -> str:
    """將分數轉成等級文字標籤。"""
    if score is None:
        return "待分析"
    if score >= 90:
        return "表現優秀"
    if score >= 75:
        return "表現良好"
    if score >= 60:
        return "尚可"
    return "需加強"


def _grade_color(score) -> str:
    """等級對應顏色。"""
    if score is None:
        return "#aeaeb2"
    if score >= 75:
        return "#34c759"
    if score >= 60:
        return "#ff9500"
    return "#ff3b30"


def _score_bar_html(score, width: int = 10, color: str = "#007aff") -> str:
    """進度條（Qt HTML ▬ 字元版）。color 決定已填充部分顏色。"""
    if score is None:
        return '<span style="color:#e5e5ea;">' + '▬' * width + '</span>'
    filled = int(round(score / 100 * width))
    return (
        f'<span style="color:{color};">' + '▬' * filled + '</span>'
        + '<span style="color:#e5e5ea;">' + '▬' * (width - filled) + '</span>'
    )


def _section_header(en: str, zh: str) -> str:
    """全大寫灰色區塊標題（稍大、帶字距感）。"""
    return (
        f'<p style="font-size:12px; font-weight:700; color:#8e8e93; letter-spacing:1.5px;'
        f' margin:18px 0 7px 0;">{en.upper()}  {zh}</p>'
    )


def _entry_html(action: str, grade: str, grade_color: str,
                ts: str, advice_str: str, extra_str: str = "") -> str:
    """單筆擊球紀錄 HTML（彩色圓點，純文字，無方形背景）。"""
    action_color = _ACTION_COLOR.get(action, "#32ade6")
    extra_part = (
        f'&nbsp;&nbsp;<span style="color:#aeaeb2; font-size:11px;">{extra_str}</span>'
        if extra_str else ''
    )
    return (
        f'<p style="margin:0 0 6px 0; padding:2px 6px; line-height:1.6;">'
        f'<span style="color:{action_color}; font-size:15px;">●</span>&nbsp;'
        f'<b style="color:#007aff; font-size:15px;">{action}</b>'
        f'&nbsp;&nbsp;<span style="color:{grade_color}; font-size:12px;'
        f' font-weight:600;">{grade}</span>'
        f'&nbsp;&nbsp;<span style="color:#aeaeb2; font-size:11px;">{ts}</span>'
        f'<br/>'
        f'&nbsp;&nbsp;&nbsp;&nbsp;'
        f'<span style="color:#6e6e73; font-size:12px;">{advice_str}</span>'
        + extra_part
        + f'</p>'
    )


def generate_html_report(event_log: list, video_name: str = "", total_ms: int = 0,
                         include_shot_log: bool = False) -> str:
    """產生 HTML 格式報告（左側彩色細線風格，無方形背景）。
    結構：摘要（大號評分）→ [擊球紀錄，僅 include_shot_log=True 時] → 各動作分析 → 最需改善
    """
    if not event_log:
        return '<p style="color:#6e6e73;">尚無分析資料。</p>'

    # ShuttleSet 12 種球種（ML 分類器），加上舊名稱退路
    action_names = list(_ACTION_COLOR.keys()) + ["高遠球", "吊球", "平抽球"]
    valid_events = [e for e in event_log if e.get("action") in action_names]
    all_scores   = [e["dtw_score"] for e in valid_events if e.get("dtw_score") is not None]
    avg_all      = sum(all_scores) / len(all_scores) if all_scores else None

    duration = ""
    if total_ms > 0:
        s = total_ms // 1000
        duration = f"{s // 60:02d}:{s % 60:02d}"

    p = []

    # ── 1. 摘要：大號居中評分 ──────────────────────────────
    avg_str   = f"{avg_all:.0f}%" if avg_all is not None else "—"
    avg_grade = _grade_label(avg_all)
    avg_color = _grade_color(avg_all)
    meta_parts = []
    if video_name:    meta_parts.append(video_name)
    if duration:      meta_parts.append(duration)
    if valid_events:  meta_parts.append(f"{len(valid_events)} 次有效擊球")
    meta = "  ·  ".join(meta_parts)

    p.append(
        f'<p style="text-align:center; margin:6px 0 2px 0;">'
        f'<span style="font-size:11px; color:#aeaeb2;">整場影片分析報告</span></p>'
        f'<p style="text-align:center; margin:0 0 2px 0;">'
        f'<span style="font-size:38px; font-weight:700; color:{avg_color};">{avg_str}</span></p>'
        f'<p style="text-align:center; margin:0 0 2px 0;">'
        f'<span style="font-size:14px; font-weight:600; color:{avg_color};">{avg_grade}</span></p>'
        f'<p style="text-align:center; margin:0 0 0 0;">'
        f'<span style="font-size:11px; color:#aeaeb2;">{meta}</span></p>'
    )

    _SEP = (
        '<table width="100%" cellpadding="0" cellspacing="0"'
        ' style="margin:18px 0 0 0; border-collapse:collapse;">'
        '<tr>'
        '<td width="10"></td>'
        '<td style="border-top:1px solid #e5e5ea;"></td>'
        '<td width="10"></td>'
        '</tr>'
        '</table>'
    )

    # ── 2. 擊球紀錄（僅匯出版本顯示）─────────────────────────
    if include_shot_log:
        p.append(_SEP)
        p.append(_section_header("SHOT LOG", "擊球紀錄"))
        for ev in event_log:
            ts         = ms_to_timestamp(ev.get("timestamp_ms", 0))
            action     = ev.get("action", "?")
            dtw_score  = ev.get("dtw_score")
            advice     = ev.get("advice", [])
            ball_speed = ev.get("ball_speed", 0.0)
            hit_height = ev.get("hit_height", 0.0)
            grade      = _grade_label(dtw_score)
            grade_color = _grade_color(dtw_score)
            advice_str = advice[0] if advice else "—"
            extra_parts = []
            if ball_speed > 0:
                extra_parts.append(f"球速 {ball_speed * BALL_SPEED_KMH_SCALE:.0f}km/h")
            if hit_height > 0:
                extra_parts.append(_hit_height_label(hit_height))
            extra_str = "  ".join(extra_parts)
            p.append(_entry_html(action, grade, grade_color, ts, advice_str, extra_str))

    # ── 3. 各動作分析（只顯示本場出現過的動作）──────────────────
    p.append(_SEP)
    p.append(_section_header("ACTION ANALYSIS", "各動作分析"))
    has_action = False
    appeared_actions = [a for a in action_names
                        if any(e.get("action") == a for e in event_log)]
    for action in appeared_actions:
        events = [e for e in event_log if e.get("action") == action]
        if not events:
            continue
        has_action   = True
        count        = len(events)
        scores       = [e["dtw_score"] for e in events if e.get("dtw_score") is not None]
        avg          = sum(scores) / len(scores) if scores else None
        grade        = _grade_label(avg)
        grade_color  = _grade_color(avg)
        sc_txt       = f"{avg:.0f}%" if avg is not None else "N/A"
        action_color = _ACTION_COLOR.get(action, "#32ade6")
        bar          = _score_bar_html(avg, color=action_color)

        p.append(
            f'<p style="margin:0 0 6px 0; padding:2px 6px; line-height:1.6;">'
            f'<span style="color:{action_color}; font-size:15px;">●</span>&nbsp;'
            f'<b style="color:{action_color}; font-size:15px;">{action}</b>'
            f'&nbsp;&nbsp;<span style="color:#aeaeb2; font-size:10px;">({count}次)</span>'
            f'<br/>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;'
            f'{bar}'
            f'&nbsp;&nbsp;<b style="color:#1c1c1e; font-size:13px;">{sc_txt}</b>'
            f'&nbsp;&nbsp;<span style="color:{grade_color}; font-size:12px;'
            f' font-weight:600;">{grade}</span>'
            f'</p>'
        )
    if not has_action:
        p.append('<p style="color:#aeaeb2; font-size:12px; padding:2px 6px;">'
                 '本場未偵測到有效動作。</p>')

    # ── 4. 改善建議 ───────────────────────────────────────
    p.append(_SEP)
    p.append(_section_header("TIPS", "改善建議"))
    _SYMBOLS = ["①", "②", "③", "④", "⑤"]
    seen, cnt = set(), 0
    for event in event_log:
        for adv in event.get("advice", []):
            if adv not in seen:
                sym = _SYMBOLS[cnt] if cnt < len(_SYMBOLS) else f"{cnt + 1}."
                p.append(
                    f'<p style="margin:0 0 5px 0; padding:2px 6px;'
                    f' color:#3c3c43; font-size:13px;">'
                    f'<span style="color:#007aff; font-weight:700;">{sym}</span> {adv}</p>'
                )
                seen.add(adv)
                cnt += 1
            if cnt >= 5:
                break
        if cnt >= 5:
            break
    if not seen:
        p.append('<p style="color:#aeaeb2; font-size:12px; padding:2px 6px;">'
                 '暫無具體建議（需 DTW 模板後才能提供詳細分析）</p>')

    return "".join(p)



def _wrap_html_page_qt(body_html: str) -> str:
    """Qt QTextEdit 版本：以完整 HTML 框架包覆，讓內容有足夠 margin 並對齊瀏覽器效果。"""
    return (
        '<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><style>'
        'body { font-family: "Segoe UI Variable","Segoe UI","PingFang TC",'
        '"Microsoft JhengHei UI",sans-serif; font-size:14px; color:#1c1c1e;'
        ' margin:0; }'
        '</style></head><body>'
        + body_html
        + '</body></html>'
    )


def _wrap_html_page(body_html: str, title: str = "羽球動作分析報告") -> str:
    """將 generate_html_report() 產生的 HTML 片段包裝成完整可獨立開啟的 HTML 頁面。"""
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{
    background: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC",
                 "Microsoft JhengHei", sans-serif;
    font-size: 15px;
    color: #1c1c1e;
    margin: 0;
    padding: 0;
  }}
  .page {{
    max-width: 680px;
    margin: 32px auto;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 2px 16px rgba(0,0,0,.08);
    padding: 28px 28px 36px;
  }}
  .page-title {{
    font-size: 13px;
    font-weight: 700;
    color: #8e8e93;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
</style>
</head>
<body>
<div class="page">
{body_html}
</div>
</body>
</html>"""
