from __future__ import annotations

from typing import List

import numpy as np
import torch


BASE_FEATURE_NAMES = [
    "u_err_near",
    "u_err_lookahead",
    "u_err_ctrl",
    "slope",
    "n_visible",
    "in_recovery",
    "vx_prev",
    "wz_prev",
]


def compute_u_err_ctrl(follower) -> torch.Tensor:
    u_err_ctrl = follower.vision_u_err
    if not getattr(follower, "use_lookahead", False):
        return u_err_ctrl

    la_alpha_normal = float(np.clip(getattr(follower, "lookahead_alpha_normal", 0.0), 0.0, 1.0))
    la_alpha_recovery = float(np.clip(getattr(follower, "lookahead_alpha_recovery", 0.0), 0.0, 1.0))
    la_alpha = torch.where(
        follower.in_recovery,
        torch.full_like(follower.vision_u_err, la_alpha_recovery),
        torch.full_like(follower.vision_u_err, la_alpha_normal),
    )
    u_blend = (1.0 - la_alpha) * follower.vision_u_err + la_alpha * follower.vision_u_err_la
    enough_pts = follower.n_visible >= 2.0
    return torch.where(enough_pts, u_blend, follower.vision_u_err)


def extract_base_features(follower) -> torch.Tensor:
    u_err_ctrl = compute_u_err_ctrl(follower)
    return torch.stack(
        [
            follower.vision_u_err,
            follower.vision_u_err_la,
            u_err_ctrl,
            follower.vision_slope,
            follower.n_visible,
            follower.in_recovery.to(torch.float32),
            follower.v_cmd_prev,
            follower.w_cmd_prev,
        ],
        dim=1,
    )


def build_feature_names(history_steps: int) -> List[str]:
    history_steps = max(1, int(history_steps))
    if history_steps == 1:
        return list(BASE_FEATURE_NAMES)
    out: List[str] = []
    for k in range(history_steps - 1, -1, -1):
        suffix = f"_t-{k}"
        out.extend([name + suffix for name in BASE_FEATURE_NAMES])
    return out


class FeatureHistoryStack:
    def __init__(self, num_envs: int, feature_dim: int, history_steps: int, device: torch.device):
        self.num_envs = int(num_envs)
        self.feature_dim = int(feature_dim)
        self.history_steps = max(1, int(history_steps))
        self.device = device
        self.buffer = torch.zeros(
            self.num_envs,
            self.history_steps,
            self.feature_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def out_dim(self) -> int:
        return self.feature_dim * self.history_steps

    def reset(self, env_ids=None):
        if env_ids is None:
            self.initialized[:] = False
            self.buffer.zero_()
            return
        if len(env_ids) == 0:
            return
        self.initialized[env_ids] = False
        self.buffer[env_ids] = 0.0

    def update(self, base_features: torch.Tensor) -> torch.Tensor:
        if base_features.shape != (self.num_envs, self.feature_dim):
            raise ValueError(
                f"Expected feature shape {(self.num_envs, self.feature_dim)}, "
                f"got {tuple(base_features.shape)}"
            )
        if self.history_steps == 1:
            return base_features

        already = self.initialized
        if torch.any(already):
            self.buffer[already, :-1, :] = self.buffer[already, 1:, :].clone()
            self.buffer[already, -1, :] = base_features[already]

        fresh = ~already
        if torch.any(fresh):
            tiled = base_features[fresh].unsqueeze(1).repeat(1, self.history_steps, 1)
            self.buffer[fresh] = tiled
            self.initialized[fresh] = True

        return self.buffer.reshape(self.num_envs, -1)

