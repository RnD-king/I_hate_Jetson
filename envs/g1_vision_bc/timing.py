DEFAULT_DECISION_HOLD_STEPS = 10


def should_update_highlevel(step_idx: int, hold_steps: int) -> bool:
    hs = max(1, int(hold_steps))
    return int(step_idx) % hs == 0

