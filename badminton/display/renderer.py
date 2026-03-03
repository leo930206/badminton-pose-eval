import cv2
import numpy as np


def draw_shuttle_trail(
    frame: np.ndarray,
    trail: list,          # [(x, y), ...] 由新到舊，None 表示該幀不可見
    trail_len: int = 10,
) -> None:
    """
    在影片畫面上繪製羽球軌跡殘影。

    Args:
        frame:     BGR 影片幀（in-place 修改）
        trail:     最近 trail_len 幀的球位置，index 0 = 最新
        trail_len: 殘影長度
    """
    for i, pos in enumerate(trail[:trail_len]):
        if pos is None:
            continue
        x, y = pos
        # 半徑與透明度隨距離遞減
        alpha  = 1.0 - i / trail_len
        radius = max(2, int(6 * alpha))
        color  = (0, int(140 * alpha), int(255 * alpha))   # 橘黃色殘影
        cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1)   # 黑色外框
        cv2.circle(frame, (x, y), radius,     color,      -1)


def draw_landmarks(image, landmarks, connections, color) -> None:
    height, width = image.shape[:2]
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(image, (x, y), 2, color, -1)
    for conn in connections:
        start = landmarks[conn.start]
        end = landmarks[conn.end]
        x1, y1 = int(start.x * width), int(start.y * height)
        x2, y2 = int(end.x * width), int(end.y * height)
        cv2.line(image, (x1, y1), (x2, y2), color, 2)


def draw_text_lines(image, lines, origin, color, scale=0.9, thickness=2) -> None:
    x, y = origin
    line_height = int(30 * scale)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
