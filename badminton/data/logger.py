import json
import os


def log_event(event_log, action, grade, context, features):
    event_log.append(
        {
            "timestamp_ms": features["timestamp_ms"],
            "action": action,
            "grade": grade,
            "context": context,
            "elbow_angle": round(features["elbow_angle"], 2),
            "wrist_speed": round(features["wrist_speed"], 3),
            "wrist_vx": round(features["wrist_vx"], 3),
            "wrist_vy": round(features["wrist_vy"], 3),
            "wrist_y": round(features["wrist_y"], 3),
        }
    )


def flush_event_log(event_log, path: str) -> None:
    if not event_log:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in event_log:
            f.write(json.dumps(item) + "\n")
