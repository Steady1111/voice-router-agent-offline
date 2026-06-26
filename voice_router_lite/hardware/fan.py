"""风扇驱动逻辑 — 5 档 PWM 调速。

从 voice-router-agent/server/intent/tools.py + orchestrator.py 移植。
直流风扇最低启动电压约 40% 占空比，PWM 起点从 102 开始。
"""

from typing import Optional

# ── 风扇 5 档位映射 ──
# 1 档 40% 最低可靠转速（微风）
# 2 档 55% 低档
# 3 档 70% 中档（默认）
# 4 档 85% 高档
# 5 档 100% 极速
SPEED_LEVELS: dict[int, int] = {
    1: 102,
    2: 140,
    3: 178,
    4: 216,
    5: 255,
}

# 档位名称
LEVEL_NAMES: dict[int, str] = {
    1: "微风",
    2: "低档",
    3: "中档",
    4: "高档",
    5: "极速",
}


def speed_level_to_pwm(level: int) -> int:
    """档位 (1-5) → PWM 值 (102-255)。"""
    return SPEED_LEVELS.get(level, 178)


def pwm_to_speed_level(pwm: int) -> int:
    """PWM 值 (0-255) → 最近档位 (1-5)。"""
    if pwm <= 0:
        return 1
    closest = 1
    min_diff = abs(SPEED_LEVELS[1] - pwm)
    for level in range(2, 6):
        diff = abs(SPEED_LEVELS[level] - pwm)
        if diff < min_diff:
            closest = level
            min_diff = diff
    return closest


def compute_relative_level(
    current_level: int,
    direction: str,
    target_level: Optional[int] = None,
) -> tuple[int, bool, str]:
    """根据方向和当前档位计算新档位。

    Returns:
        (new_level, at_limit, limit_msg)
    """
    if direction == "up":
        if current_level >= 5:
            return (5, True, "已经是最快档了")
        return (current_level + 1, False, "")
    elif direction == "down":
        if current_level <= 1:
            return (1, True, "已经是最低档了")
        return (current_level - 1, False, "")
    elif direction == "max":
        return (5, False, "")
    elif direction == "min":
        return (1, False, "")
    elif direction == "set" and target_level is not None:
        new_level = max(1, min(5, target_level))
        at_limit = (new_level == current_level and new_level in (1, 5))
        limit_msg = ""
        if at_limit and new_level == 5:
            limit_msg = "已经是最快档了"
        elif at_limit and new_level == 1:
            limit_msg = "已经是最低档了"
        return (new_level, at_limit, limit_msg)
    else:
        return (current_level, False, "")
