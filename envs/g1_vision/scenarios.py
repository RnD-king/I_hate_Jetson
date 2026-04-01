from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EpisodeScenario:
    scenario_id: str
    preset: str
    level: str
    path_blocks: List[Tuple]
    manual_straight_len_jitter: float
    manual_turn_radius_jitter: float
    manual_turn_angle_jitter_deg: float
    per_pt_dropout_prob: float
    burst_dropout_prob: float
    pixel_jitter_std: float
    straight_wiggle_amp_min: float
    straight_wiggle_amp_max: float
    straight_wiggle_freq_min: float
    straight_wiggle_freq_max: float
    turn_wiggle_amp_min: float
    turn_wiggle_amp_max: float
    turn_wiggle_freq_min: float
    turn_wiggle_freq_max: float
    init_y_range: float
    init_yaw_range: float

    def to_dict(self) -> Dict:
        return asdict(self)


PRESET_LEVEL_WEIGHTS = {
    "basic": (("basic", 1.0), ("medium", 0.0), ("hard", 0.00), ("extreme", 0.0)),
    "mixed": (("basic", 0.35), ("medium", 0.45), ("hard", 0.20), ("extreme", 0.0)),
    "hard": (("basic", 0.05), ("medium", 0.35), ("hard", 0.60), ("extreme", 0.0)),
    "extreme": (("basic", 0.0), ("medium", 0.0), ("hard", 0.0), ("extreme", 1.0)),
}


LEVEL_RANGES = {
    "basic": {
        "num_turns": (2, 4),
        "straight_len": (3.0, 7.5),
        "radius": (1.6, 2.8),
        "angle_deg": (60.0, 90.0),
        "manual_straight_len_jitter": (0.00, 0.20),
        "manual_turn_radius_jitter": (0.00, 0.10),
        "manual_turn_angle_jitter_deg": (0.0, 0.0),
        "per_pt_dropout_prob": (0.02, 0.10),
        "burst_dropout_prob": (0.00, 0.04),
        "pixel_jitter_std": (0.5, 3.0),
        "straight_wiggle_amp": (0.04, 0.14),
        "straight_wiggle_freq": (2.0, 4.0),
        "turn_wiggle_amp": (0.02, 0.08),
        "turn_wiggle_freq": (1.5, 3.0),
        "init_y_range": (0.02, 0.12),
        "init_yaw_range": (0.05, 0.20),
    },
    "medium": {
        "num_turns": (3, 6),
        "straight_len": (2.0, 6.5),
        "radius": (1.2, 2.2),
        "angle_deg": (60.0, 90.0),
        "manual_straight_len_jitter": (0.10, 0.40),
        "manual_turn_radius_jitter": (0.05, 0.20),
        "manual_turn_angle_jitter_deg": (0.0, 0.0),
        "per_pt_dropout_prob": (0.08, 0.20),
        "burst_dropout_prob": (0.02, 0.08),
        "pixel_jitter_std": (2.0, 6.0),
        "straight_wiggle_amp": (0.10, 0.24),
        "straight_wiggle_freq": (2.5, 5.5),
        "turn_wiggle_amp": (0.05, 0.14),
        "turn_wiggle_freq": (2.0, 5.0),
        "init_y_range": (0.08, 0.20),
        "init_yaw_range": (0.10, 0.35),
    },
    "hard": {
        "num_turns": (5, 9),
        "straight_len": (1.4, 5.2),
        "radius": (0.8, 1.8),
        "angle_deg": (60.0, 90.0),
        "manual_straight_len_jitter": (0.20, 0.65),
        "manual_turn_radius_jitter": (0.10, 0.35),
        "manual_turn_angle_jitter_deg": (0.0, 0.0),
        "per_pt_dropout_prob": (0.15, 0.35),
        "burst_dropout_prob": (0.06, 0.16),
        "pixel_jitter_std": (4.0, 11.0),
        "straight_wiggle_amp": (0.16, 0.35),
        "straight_wiggle_freq": (3.0, 7.0),
        "turn_wiggle_amp": (0.08, 0.20),
        "turn_wiggle_freq": (3.0, 7.5),
        "init_y_range": (0.16, 0.30),
        "init_yaw_range": (0.25, 0.55),
    },
    "extreme": {
        "num_turns": (7, 9),
        "straight_len": (1.4, 3.2),
        "radius": (0.8, 1.3),
        "angle_deg": (89.9, 90.0),
        "manual_straight_len_jitter": (0.20, 0.65),
        "manual_turn_radius_jitter": (0.10, 0.35),
        "manual_turn_angle_jitter_deg": (0.0, 0.0),
        "per_pt_dropout_prob": (0.15, 0.35),
        "burst_dropout_prob": (0.06, 0.16),
        "pixel_jitter_std": (4.0, 11.0),
        "straight_wiggle_amp": (0.16, 0.25),
        "straight_wiggle_freq": (7.0, 12.0),
        "turn_wiggle_amp": (0.08, 0.20),
        "turn_wiggle_freq": (5.0, 7.5),
        "init_y_range": (0.16, 0.30),
        "init_yaw_range": (0.25, 0.55),
    },
}


def _sample_uniform(rng: np.random.Generator, lo_hi: Tuple[float, float]) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def _sample_int(rng: np.random.Generator, lo_hi: Tuple[int, int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi + 1))


def _sample_level(rng: np.random.Generator, preset: str) -> str:
    if preset not in PRESET_LEVEL_WEIGHTS:
        raise ValueError(
            f"Unknown scenario_preset '{preset}'. "
            f"Use one of {list(PRESET_LEVEL_WEIGHTS.keys())}."
        )
    levels, probs = zip(*PRESET_LEVEL_WEIGHTS[preset])
    probs_np = np.asarray(probs, dtype=np.float64)
    probs_np = probs_np / np.sum(probs_np)
    idx = int(rng.choice(len(levels), p=probs_np))
    return levels[idx]


def _sample_path_blocks(level_cfg: Dict, rng: np.random.Generator) -> List[Tuple]:
    num_turns = _sample_int(rng, level_cfg["num_turns"])
    straight_lo, straight_hi = level_cfg["straight_len"]
    radius_lo, radius_hi = level_cfg["radius"]
    angle_lo, angle_hi = level_cfg["angle_deg"]

    blocks: List[Tuple] = []
    blocks.append(("S", _sample_uniform(rng, (straight_lo * 1.2, straight_hi * 1.4))))
    for _ in range(num_turns):
        turn_kind = "L" if rng.uniform() < 0.5 else "R"
        radius = _sample_uniform(rng, (radius_lo, radius_hi))
        angle_deg = _sample_uniform(rng, (angle_lo, angle_hi))
        blocks.append((turn_kind, radius, angle_deg))
        blocks.append(("S", _sample_uniform(rng, (straight_lo, straight_hi))))
    blocks.append(("S", _sample_uniform(rng, (straight_lo * 0.8, straight_hi * 1.2))))
    return blocks


def sample_episode_scenario(
    rng: np.random.Generator,
    preset: str,
    episode_idx: int,
) -> EpisodeScenario:
    level = _sample_level(rng, preset=preset)
    cfg = LEVEL_RANGES[level]
    path_blocks = _sample_path_blocks(cfg, rng)

    scenario_id = f"{preset}_{level}_ep{episode_idx:06d}_{int(rng.integers(1_000_000_000)):09d}"

    return EpisodeScenario(
        scenario_id=scenario_id,
        preset=preset,
        level=level,
        path_blocks=path_blocks,
        manual_straight_len_jitter=_sample_uniform(rng, cfg["manual_straight_len_jitter"]),
        manual_turn_radius_jitter=_sample_uniform(rng, cfg["manual_turn_radius_jitter"]),
        manual_turn_angle_jitter_deg=_sample_uniform(rng, cfg["manual_turn_angle_jitter_deg"]),
        per_pt_dropout_prob=_sample_uniform(rng, cfg["per_pt_dropout_prob"]),
        burst_dropout_prob=_sample_uniform(rng, cfg["burst_dropout_prob"]),
        pixel_jitter_std=_sample_uniform(rng, cfg["pixel_jitter_std"]),
        straight_wiggle_amp_min=_sample_uniform(
            rng, (cfg["straight_wiggle_amp"][0], cfg["straight_wiggle_amp"][1] * 0.9)
        ),
        straight_wiggle_amp_max=_sample_uniform(rng, cfg["straight_wiggle_amp"]),
        straight_wiggle_freq_min=_sample_uniform(
            rng, (cfg["straight_wiggle_freq"][0], cfg["straight_wiggle_freq"][1] * 0.8)
        ),
        straight_wiggle_freq_max=_sample_uniform(rng, cfg["straight_wiggle_freq"]),
        turn_wiggle_amp_min=_sample_uniform(
            rng, (cfg["turn_wiggle_amp"][0], cfg["turn_wiggle_amp"][1] * 0.8)
        ),
        turn_wiggle_amp_max=_sample_uniform(rng, cfg["turn_wiggle_amp"]),
        turn_wiggle_freq_min=_sample_uniform(
            rng, (cfg["turn_wiggle_freq"][0], cfg["turn_wiggle_freq"][1] * 0.75)
        ),
        turn_wiggle_freq_max=_sample_uniform(rng, cfg["turn_wiggle_freq"]),
        init_y_range=_sample_uniform(rng, cfg["init_y_range"]),
        init_yaw_range=_sample_uniform(rng, cfg["init_yaw_range"]),
    )


def apply_scenario_to_follower(follower, scenario: EpisodeScenario):
    """
    Apply one scenario to follower.
    For now dataset scripts default to num_envs=1, so global perception/control
    parameters are applied uniformly.
    """

    follower.path_mode = "block"
    follower.manual_block_specs = [list(scenario.path_blocks)]

    follower.manual_straight_len_jitter = float(scenario.manual_straight_len_jitter)
    follower.manual_turn_radius_jitter = float(scenario.manual_turn_radius_jitter)
    follower.manual_turn_angle_jitter_deg = float(scenario.manual_turn_angle_jitter_deg)

    follower.per_pt_dropout_prob = float(scenario.per_pt_dropout_prob)
    follower.burst_dropout_prob = float(scenario.burst_dropout_prob)
    follower.pixel_jitter_std = float(scenario.pixel_jitter_std)

    follower.straight_wiggle_amp_min = float(
        min(scenario.straight_wiggle_amp_min, scenario.straight_wiggle_amp_max)
    )
    follower.straight_wiggle_amp_max = float(
        max(scenario.straight_wiggle_amp_min, scenario.straight_wiggle_amp_max)
    )
    follower.straight_wiggle_freq_min = float(
        min(scenario.straight_wiggle_freq_min, scenario.straight_wiggle_freq_max)
    )
    follower.straight_wiggle_freq_max = float(
        max(scenario.straight_wiggle_freq_min, scenario.straight_wiggle_freq_max)
    )
    follower.turn_wiggle_amp_min = float(min(scenario.turn_wiggle_amp_min, scenario.turn_wiggle_amp_max))
    follower.turn_wiggle_amp_max = float(max(scenario.turn_wiggle_amp_min, scenario.turn_wiggle_amp_max))
    follower.turn_wiggle_freq_min = float(
        min(scenario.turn_wiggle_freq_min, scenario.turn_wiggle_freq_max)
    )
    follower.turn_wiggle_freq_max = float(
        max(scenario.turn_wiggle_freq_min, scenario.turn_wiggle_freq_max)
    )

    follower.setup_random_dotted_spline_paths()
