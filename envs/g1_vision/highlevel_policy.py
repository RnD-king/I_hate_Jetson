from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class HighLevelMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 128), output_dim: int = 2):
        super().__init__()
        dims = [int(input_dim)] + [int(h) for h in hidden_dims]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (features - mean) / torch.clamp(std, min=eps)


def parse_hidden_dims(spec: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(spec).split(",")]
    dims = [int(p) for p in parts if p]
    if len(dims) == 0:
        return (128, 128)
    return tuple(dims)


def load_bc_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = int(ckpt["input_dim"])
    hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", (128, 128)))
    output_dim = int(ckpt.get("output_dim", 2))

    model = HighLevelMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    feature_mean = torch.tensor(ckpt["feature_mean"], device=device, dtype=torch.float32)
    feature_std = torch.tensor(ckpt["feature_std"], device=device, dtype=torch.float32)
    feature_names = list(ckpt.get("feature_names", []))

    return model, feature_mean, feature_std, feature_names, ckpt


class HighLevelCommandAdapter:
    """
    Keeps the same command timing/smoothing envelope used by the teacher follower:
    - hold update every `follower.hold_steps`
    - rate limit with `follower._clip_and_rate_limit`
    - interpolate over `follower.command_interp_steps`
    """

    def __init__(self, follower):
        self.follower = follower

    def step(self, target_vx_wz: torch.Tensor, step_idx: int) -> torch.Tensor:
        if target_vx_wz.shape[0] != self.follower.num_envs or target_vx_wz.shape[1] != 2:
            raise ValueError(
                f"Expected target_vx_wz shape [{self.follower.num_envs}, 2], "
                f"got {tuple(target_vx_wz.shape)}"
            )

        target = target_vx_wz.to(self.follower.device)
        interp_steps = max(1, int(self.follower.command_interp_steps))

        if step_idx % int(self.follower.hold_steps) == 0:
            v_raw = target[:, 0]
            w_raw = target[:, 1]
            v_cmd, w_cmd = self.follower._clip_and_rate_limit(v_raw=v_raw, w_raw=w_raw)

            self.follower.v_cmd_start = self.follower.v_cmd_hold.clone()
            self.follower.w_cmd_start = self.follower.w_cmd_hold.clone()
            self.follower.v_cmd_goal = v_cmd
            self.follower.w_cmd_goal = w_cmd
            self.follower.interp_countdown[:] = interp_steps

        if interp_steps <= 1:
            self.follower.v_cmd_hold = self.follower.v_cmd_goal
            self.follower.w_cmd_hold = self.follower.w_cmd_goal
            self.follower.interp_countdown.zero_()
        else:
            active = self.follower.interp_countdown > 0
            alpha = (
                (interp_steps - self.follower.interp_countdown + 1).to(torch.float32)
                / float(interp_steps)
            )
            alpha = torch.clamp(alpha, 0.0, 1.0)

            v_interp = self.follower.v_cmd_start + (self.follower.v_cmd_goal - self.follower.v_cmd_start) * alpha
            w_interp = self.follower.w_cmd_start + (self.follower.w_cmd_goal - self.follower.w_cmd_start) * alpha

            self.follower.v_cmd_hold = torch.where(active, v_interp, self.follower.v_cmd_goal)
            self.follower.w_cmd_hold = torch.where(active, w_interp, self.follower.w_cmd_goal)
            self.follower.interp_countdown = torch.where(
                active, self.follower.interp_countdown - 1, self.follower.interp_countdown
            )

        self.follower.v_cmd_prev = self.follower.v_cmd_hold
        self.follower.w_cmd_prev = self.follower.w_cmd_hold

        commands = torch.zeros((self.follower.num_envs, 4), device=self.follower.device)
        commands[:, 0] = self.follower.v_cmd_hold
        commands[:, 1] = 0.0
        commands[:, 2] = self.follower.w_cmd_hold
        commands[:, 3] = 0.0
        return commands
