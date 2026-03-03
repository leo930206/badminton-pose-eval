import cv2


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
