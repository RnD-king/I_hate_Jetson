from __future__ import annotations

import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def compute_path_metrics(
    follower,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    yaw: torch.Tensor,
):
    """
    Computes path progress ratio, nearest distance to path, and heading error
    for each env based on follower's cached path tensors.
    """
    device = local_x.device
    num_envs = int(local_x.shape[0])

    path_ratio = torch.zeros(num_envs, device=device, dtype=torch.float32)
    path_dist = torch.full((num_envs,), 1e9, device=device, dtype=torch.float32)
    heading_err = torch.zeros(num_envs, device=device, dtype=torch.float32)

    for i in range(num_envs):
        path_xy = follower.path_points[i]
        path_s = follower.path_s[i]
        path_heading = follower.path_heading[i]
        if path_xy is None or path_s is None or path_heading is None or path_xy.shape[0] == 0:
            continue

        pos = torch.stack([local_x[i], local_y[i]], dim=0)
        delta = path_xy - pos[None, :]
        d2 = torch.sum(delta * delta, dim=1)

        nearest_idx = torch.argmin(d2)
        dmin = torch.sqrt(torch.clamp(d2[nearest_idx], min=0.0))

        s_now = path_s[nearest_idx]
        s_end = torch.clamp(path_s[-1], min=1e-6)
        ratio = torch.clamp(s_now / s_end, 0.0, 1.0)

        yaw_ref = path_heading[nearest_idx]
        yaw_err = wrap_to_pi(yaw[i] - yaw_ref)

        path_ratio[i] = ratio
        path_dist[i] = dmin
        heading_err[i] = yaw_err

    return path_ratio, path_dist, heading_err
