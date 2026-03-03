def estimate_context(features) -> str:
    if features["wrist_above_head"] and features["wrist_speed"] > 1.4:
        return "offense"
    if features["body_speed"] > 0.8 and not features["wrist_above_shoulder"]:
        return "defense"
    return "neutral"
