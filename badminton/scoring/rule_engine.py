from config import Config


def grade_action(action, features, context, max_wrist_y_in_prep, config: Config):
    angle = features["elbow_angle"]
    advice = []
    grade = "Good"

    if action == "Smash":
        if angle < config.elbow_feedback_threshold:
            grade = "Needs Work"
            advice.append("Elbow too bent. Extend your arm for a longer swing.")
        elif angle >= config.perfect_angle_min and features["wrist_y"] <= max_wrist_y_in_prep + 0.02:
            grade = "Excellent"
            advice.append("Perfect smash! Excellent contact point.")
        else:
            advice.append("Straighten your elbow a bit more.")

    elif action == "Clear":
        if angle >= config.perfect_angle_min and features["wrist_above_head"]:
            grade = "Excellent"
            advice.append("Strong clear with good extension.")
        elif angle < config.elbow_feedback_threshold:
            grade = "Needs Work"
            advice.append("Extend the elbow more for a higher clear.")
        else:
            advice.append("Increase upward speed for more height.")

    elif action == "Drop":
        if features["wrist_speed"] < config.min_down_speed:
            grade = "Excellent"
            advice.append("Soft drop with good control.")
        else:
            grade = "Good"
            advice.append("Soften the wrist speed for a tighter drop.")

    elif action == "Drive":
        if 150 <= angle <= 175 and abs(features["wrist_vx"]) > config.min_horizontal_speed:
            grade = "Excellent"
            advice.append("Fast drive with solid extension.")
        else:
            grade = "Needs Work"
            advice.append("Keep the elbow more extended for a flatter drive.")

    elif action == "Cut":
        if 140 <= angle <= 165 and features["wrist_vy"] > config.min_drop_speed:
            grade = "Good"
            advice.append("Nice cut angle.")
        else:
            grade = "Needs Work"
            advice.append("Sharpen the cut with a quicker downward snap.")

    if context == "defense" and grade == "Needs Work":
        advice.append("Defensive position detected; focus on control and recovery.")
    if context == "neutral" and not advice:
        advice.append("Maintain balance and rhythm.")

    return grade, advice[:3]
