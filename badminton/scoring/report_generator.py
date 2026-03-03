"""
整場分析報告產生器
用途：把分析過程中累積的 event_log 轉成人類可讀的文字報告。
"""


def ms_to_timestamp(ms: int) -> str:
    """將毫秒轉成 MM:SS 格式。"""
    total_sec = ms // 1000
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"


def score_to_stars(score) -> str:
    """將 0~100 分數轉成星等。"""
    if score is None:
        return "──────"
    if score >= 90:
        return "★★★★★"
    if score >= 75:
        return "★★★★☆"
    if score >= 60:
        return "★★★☆☆"
    if score >= 45:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def score_to_bar(score, width: int = 14) -> str:
    """將分數轉成文字進度條。"""
    if score is None:
        return "░" * width + "  N/A"
    filled = int(round(score / 100 * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {score:.0f}%"


def _status_label(avg_score) -> str:
    if avg_score is None:
        return "尚無 DTW 模板資料"
    if avg_score >= 75:
        return "✓ 表現良好"
    if avg_score >= 60:
        return "△ 尚可，有進步空間"
    return "⚠ 需要加強"


def generate_report(event_log: list, video_name: str = "", total_ms: int = 0) -> str:
    """
    根據 event_log 產生完整的文字報告。

    event_log 中每筆 dict 包含：
        timestamp_ms, action, grade, context,
        dtw_score (可為 None), advice (list)
    """
    if not event_log:
        return "尚無分析資料。"

    action_names = ["Smash", "Clear", "Drop", "Drive", "Cut"]
    lines = []

    # 標題
    lines.append("整場影片分析報告")
    lines.append("═" * 51)

    if video_name:
        lines.append(f"影片：{video_name}")

    if total_ms > 0:
        total_sec = total_ms // 1000
        duration = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
        lines.append(f"影片長度：{duration}")

    valid_events = [e for e in event_log if e.get("action") in action_names]
    skipped = len(event_log) - len(valid_events)

    lines.append(f"有效擊球數：{len(valid_events)} 球   不列入評分：{skipped} 球")
    lines.append("")

    # 逐球紀錄
    for event in event_log:
        ts = ms_to_timestamp(event.get("timestamp_ms", 0))
        action = event.get("action", "?")
        dtw_score = event.get("dtw_score")
        advice = event.get("advice", [])

        stars = score_to_stars(dtw_score)
        score_str = f"{dtw_score:.0f}%" if dtw_score is not None else "  N/A"
        advice_str = advice[0] if advice else ""

        lines.append(f"[{ts}] {action:<6} {stars}  {score_str:<6}  {advice_str}")

    lines.append("")
    lines.append("─" * 51)
    lines.append("各動作平均分數")
    lines.append("")

    # 統計各動作
    worst_action = None
    worst_score = float("inf")
    best_action = None
    best_score = -1.0

    for action in action_names:
        events = [e for e in event_log if e.get("action") == action]
        if not events:
            continue

        scores = [e["dtw_score"] for e in events if e.get("dtw_score") is not None]
        avg = sum(scores) / len(scores) if scores else None
        bar = score_to_bar(avg)
        status = _status_label(avg)
        lines.append(f"  {action:<8} {bar}   {status}")

        if avg is not None:
            if avg < worst_score:
                worst_score = avg
                worst_action = action
            if avg > best_score:
                best_score = avg
                best_action = action

    lines.append("")
    lines.append("─" * 51)

    # 最需改善
    lines.append("本場最需改善的地方")
    lines.append("")
    advice_count = 0
    seen_advice = set()
    for event in event_log:
        for adv in event.get("advice", []):
            if adv not in seen_advice:
                advice_count += 1
                lines.append(f"  {advice_count}. {adv}")
                seen_advice.add(adv)
            if advice_count >= 3:
                break
        if advice_count >= 3:
            break

    if not seen_advice:
        lines.append("  暫無具體建議（需要 DTW 模板才能提供詳細分析）")

    lines.append("")
    lines.append("─" * 51)

    # 表現最好的動作
    if best_action:
        lines.append(f"本場表現最好的動作")
        lines.append("")
        lines.append(f"  {best_action}：平均相似度 {best_score:.0f}%，{'接近職業標準' if best_score >= 75 else '仍有進步空間'}")

    lines.append("")
    lines.append("═" * 51)

    return "\n".join(lines)
