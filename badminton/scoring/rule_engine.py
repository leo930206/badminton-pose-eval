from config import Config


def grade_action(action, features, context, max_wrist_y_in_prep, config: Config):
    angle = features["elbow_angle"]
    advice = []
    grade = "Good"

    if action == "殺球":
        if angle < config.elbow_feedback_threshold:
            grade = "Needs Work"
            advice.append("手肘角度太小，打開手臂以延長揮拍弧度")
        elif angle >= config.perfect_angle_min and features["wrist_y"] <= max_wrist_y_in_prep + 0.02:
            grade = "Excellent"
            advice.append("完美殺球！擊球點非常好")
        else:
            advice.append("手肘再打開一點")

    elif action == "高遠球":
        if angle >= config.perfect_angle_min and features["wrist_above_head"]:
            grade = "Excellent"
            advice.append("高遠球力道充足，手臂伸展良好")
        elif angle < config.elbow_feedback_threshold:
            grade = "Needs Work"
            advice.append("打開手肘以打出更高的高遠球")
        else:
            advice.append("加大向上揮速，增加球的高度")

    elif action == "吊球":
        if features["wrist_speed"] < config.min_down_speed:
            grade = "Excellent"
            advice.append("吊球輕柔，控球佳")
        else:
            grade = "Good"
            advice.append("放慢手腕速度，讓落點更精準")

    elif action == "平抽球":
        if 150 <= angle <= 175 and abs(features["wrist_vx"]) > config.min_horizontal_speed:
            grade = "Excellent"
            advice.append("平抽球速度快，手臂伸展有力")
        else:
            grade = "Needs Work"
            advice.append("保持手肘伸展，打出更平的平抽球")

    elif action == "切球":
        if 140 <= angle <= 165 and features["wrist_vy"] > config.min_drop_speed:
            grade = "Good"
            advice.append("切球角度不錯")
        else:
            grade = "Needs Work"
            advice.append("加快向下甩腕速度，切球更銳利")

    if context == "defense" and grade == "Needs Work":
        advice.append("偵測到防守姿態，專注控球與回位")
    if context == "neutral" and not advice:
        advice.append("保持平衡與節奏")

    return grade, advice[:3]
