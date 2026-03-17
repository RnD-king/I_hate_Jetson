import isaacgym

import math
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class BezierPathFollower:
    """
    anchor point들을 piecewise cubic Bézier로 연결한 경로를 추종하는 휴리스틱 상위 제어기.

    구조:
    - local frame에서 anchor point 5개 정의
    - anchor들을 모두 지나는 piecewise cubic Bézier path 생성
    - path를 샘플링한 polyline 위에서 nearest + lookahead 계산
    - target point를 향해 (vx, 0, wz) 생성
    - recovery hysteresis 포함
    - 동적 lookahead 포함
    """

    def __init__(self, num_envs, device, env_dt):
        self.num_envs = num_envs
        self.device = device
        self.env_dt = env_dt

        # ===== upper-policy safe envelope =====
        self.vx_min = 0.10
        self.vx_max = 1.5
        self.wz_min = -0.9
        self.wz_max = 0.9

        # ===== update rate =====
        self.hold_steps = 5

        # ===== rate limit =====
        self.dv_max = 0.15
        self.dw_max = 0.15

        # ===== nominal control =====
        self.v_base = 1.0
        self.k_alpha = 2.2
        self.k_yaw_damp = 0.0
        self.k_v_alpha = 0.6
        self.k_v_ey = 0.15

        # ===== dynamic lookahead =====
        self.lookahead_base = 0.30
        self.lookahead_kv = 0.20
        self.lookahead_ka = 0.20
        self.lookahead_min = 0.25
        self.lookahead_max = 0.45

        # env별 현재 lookahead 저장
        self.current_lookahead = torch.full(
            (self.num_envs,),
            self.lookahead_base,
            device=self.device
        )

        # ===== recovery hysteresis =====
        self.recover_enter_ey = 0.35
        self.recover_enter_epsi = 0.35
        self.recover_exit_ey = 0.18
        self.recover_exit_epsi = 0.15
        self.recover_vx = 0.10
        self.recover_wz = 0.9

        # ===== 내부 상태 =====
        self.v_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self.v_cmd_hold = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_hold = torch.zeros(self.num_envs, device=self.device)
        self.in_recovery = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # ===== path data =====
        self.anchor_points = None          # [E, K, 2]
        self.path_points = None            # [E, N, 2]
        self.path_s = None                 # [E, N]

        # 현재 lookahead target 저장(시각화용)
        self.target_x = torch.zeros(self.num_envs, device=self.device)
        self.target_y = torch.zeros(self.num_envs, device=self.device)

        # 최근접점도 저장(디버깅용)
        self.nearest_x = torch.zeros(self.num_envs, device=self.device)
        self.nearest_y = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------
    # Bézier / Path generation
    # ------------------------------------------------------------
    def _bezier_cubic(self, p0, c1, c2, p3, t):
        omt = 1.0 - t
        return (
            (omt ** 3).unsqueeze(-1) * p0
            + (3.0 * (omt ** 2) * t).unsqueeze(-1) * c1
            + (3.0 * omt * (t ** 2)).unsqueeze(-1) * c2
            + (t ** 3).unsqueeze(-1) * p3
        )

    def _compute_tangents(self, pts):
        K = pts.shape[0]
        tangents = torch.zeros_like(pts)

        tangents[0] = pts[1] - pts[0]
        tangents[K - 1] = pts[K - 1] - pts[K - 2]

        for i in range(1, K - 1):
            tangents[i] = 0.5 * (pts[i + 1] - pts[i - 1])

        return tangents

    def _build_piecewise_bezier_samples(self, pts, samples_per_seg=30):
        K = pts.shape[0]
        tangents = self._compute_tangents(pts)

        sampled = []

        for i in range(K - 1):
            p0 = pts[i]
            p3 = pts[i + 1]
            m0 = tangents[i]
            m1 = tangents[i + 1]

            # Hermite -> Bézier
            c1 = p0 + m0 / 3.0
            c2 = p3 - m1 / 3.0

            t = torch.linspace(0.0, 1.0, samples_per_seg, device=self.device)
            seg = self._bezier_cubic(p0, c1, c2, p3, t)

            if i > 0:
                seg = seg[1:]

            sampled.append(seg)

        samples = torch.cat(sampled, dim=0)

        diffs = samples[1:] - samples[:-1]
        seg_len = torch.sqrt(torch.sum(diffs * diffs, dim=1) + 1e-8)
        s_accum = torch.zeros(samples.shape[0], device=self.device)
        s_accum[1:] = torch.cumsum(seg_len, dim=0)

        return samples, s_accum

    def setup_anchor_paths(self, anchor_mode="split_lr"):
        left_pts = torch.tensor(
            [
                [0.30, 0.00],
                [1.70, 1.45],
                [3.10, 0.18],
                [4.55, 1.38],
                [6.05, 0.68],
            ],
            device=self.device,
            dtype=torch.float,
        )

        right_pts = torch.tensor(
            [
                [0.30, 0.00],
                [1.70, -1.45],
                [3.10, -0.18],
                [4.55, -1.38],
                [6.05, -0.68],
            ],
            device=self.device,
            dtype=torch.float,
        )

        anchor_sets = []

        if anchor_mode == "left_only":
            for _ in range(self.num_envs):
                anchor_sets.append(left_pts.clone())
        elif anchor_mode == "right_only":
            for _ in range(self.num_envs):
                anchor_sets.append(right_pts.clone())
        elif anchor_mode == "split_lr":
            half = self.num_envs // 2
            for i in range(self.num_envs):
                if i < half:
                    anchor_sets.append(left_pts.clone())
                else:
                    anchor_sets.append(right_pts.clone())
        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

        self.anchor_points = torch.stack(anchor_sets, dim=0)

        path_points = []
        path_s = []

        for i in range(self.num_envs):
            samples, s_accum = self._build_piecewise_bezier_samples(
                self.anchor_points[i],
                samples_per_seg=35
            )
            path_points.append(samples)
            path_s.append(s_accum)

        self.path_points = torch.stack(path_points, dim=0)
        self.path_s = torch.stack(path_s, dim=0)

    # ------------------------------------------------------------
    # Control logic
    # ------------------------------------------------------------
    def _clip_and_rate_limit(self, v_raw, w_raw):
        v = torch.clamp(v_raw, self.vx_min, self.vx_max)
        w = torch.clamp(w_raw, self.wz_min, self.wz_max)

        v = torch.clamp(v, self.v_cmd_prev - self.dv_max, self.v_cmd_prev + self.dv_max)
        w = torch.clamp(w, self.w_cmd_prev - self.dw_max, self.w_cmd_prev + self.dw_max)

        return v, w

    def _update_recovery_state(self, e_y, e_psi):
        enter_mask = (torch.abs(e_y) > self.recover_enter_ey) | (torch.abs(e_psi) > self.recover_enter_epsi)
        exit_mask = (torch.abs(e_y) < self.recover_exit_ey) & (torch.abs(e_psi) < self.recover_exit_epsi)

        self.in_recovery = torch.where(
            enter_mask,
            torch.ones_like(self.in_recovery),
            self.in_recovery
        )
        self.in_recovery = torch.where(
            exit_mask,
            torch.zeros_like(self.in_recovery),
            self.in_recovery
        )

    def _compute_dynamic_lookahead(self, alpha):
        """
        속도가 빠르면 늘리고, alpha가 크면 줄이는 동적 lookahead
        """
        lookahead = (
            self.lookahead_base
            + self.lookahead_kv * self.v_cmd_prev
            - self.lookahead_ka * torch.abs(alpha)
        )

        lookahead = torch.clamp(
            lookahead,
            self.lookahead_min,
            self.lookahead_max
        )

        self.current_lookahead = lookahead
        return lookahead

    def _compute_path_geometry(self, local_x, local_y, yaw):
        E = self.num_envs
        N = self.path_points.shape[1]

        robot_xy = torch.stack([local_x, local_y], dim=1)  # [E, 2]

        # 1차 nearest search
        diff = self.path_points - robot_xy.unsqueeze(1)
        dist2 = torch.sum(diff * diff, dim=2)
        nearest_idx = torch.argmin(dist2, dim=1)

        nearest_pts = self.path_points[torch.arange(E, device=self.device), nearest_idx]
        nearest_s = self.path_s[torch.arange(E, device=self.device), nearest_idx]

        self.nearest_x = nearest_pts[:, 0]
        self.nearest_y = nearest_pts[:, 1]

        # tangent at nearest
        prev_idx = torch.clamp(nearest_idx - 1, 0, N - 1)
        next_idx = torch.clamp(nearest_idx + 1, 0, N - 1)

        prev_pts = self.path_points[torch.arange(E, device=self.device), prev_idx]
        next_pts = self.path_points[torch.arange(E, device=self.device), next_idx]

        tangent = next_pts - prev_pts
        tangent_heading = torch.atan2(tangent[:, 1], tangent[:, 0])

        e_psi = tangent_heading - yaw
        e_psi = torch.atan2(torch.sin(e_psi), torch.cos(e_psi))

        tangent_norm = torch.sqrt(torch.sum(tangent * tangent, dim=1) + 1e-8)
        tangent_unit = tangent / tangent_norm.unsqueeze(1)
        normal = torch.stack([-tangent_unit[:, 1], tangent_unit[:, 0]], dim=1)

        rel = robot_xy - nearest_pts
        signed_ey = torch.sum(rel * normal, dim=1)

        # 먼저 nearest 기반 임시 alpha 계산
        temp_target_heading = torch.atan2(
            nearest_pts[:, 1] - local_y,
            nearest_pts[:, 0] - local_x
        )
        temp_alpha = temp_target_heading - yaw
        temp_alpha = torch.atan2(torch.sin(temp_alpha), torch.cos(temp_alpha))

        # 동적 lookahead 계산
        dynamic_lookahead = self._compute_dynamic_lookahead(temp_alpha)

        # 진짜 lookahead target
        target_s = nearest_s + dynamic_lookahead

        target_idx = []
        for i in range(E):
            idx = torch.searchsorted(self.path_s[i], target_s[i], right=False)
            idx = torch.clamp(idx, 0, N - 1)
            target_idx.append(idx)
        target_idx = torch.stack(target_idx, dim=0)

        target_pts = self.path_points[torch.arange(E, device=self.device), target_idx]
        self.target_x = target_pts[:, 0]
        self.target_y = target_pts[:, 1]

        target_heading = torch.atan2(target_pts[:, 1] - local_y, target_pts[:, 0] - local_x)
        alpha = target_heading - yaw
        alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))

        return nearest_pts, target_pts, signed_ey, e_psi, alpha

    def compute_commands(self, step_idx, local_x, local_y, yaw):
        if step_idx % self.hold_steps == 0:
            _, _, signed_ey, e_psi, alpha = self._compute_path_geometry(local_x, local_y, yaw)

            self._update_recovery_state(signed_ey, e_psi)

            # normal mode
            w_raw = self.k_alpha * alpha - self.k_yaw_damp * yaw
            v_raw = (
                self.v_base
                - self.k_v_alpha * torch.abs(alpha)
                - self.k_v_ey * torch.abs(signed_ey)
            )

            # recovery mode
            recover_v_raw = torch.full_like(v_raw, self.recover_vx)
            recover_w_raw = self.recover_wz * torch.sign(alpha)

            v_raw = torch.where(self.in_recovery, recover_v_raw, v_raw)
            w_raw = torch.where(self.in_recovery, recover_w_raw, w_raw)

            v_cmd, w_cmd = self._clip_and_rate_limit(v_raw, w_raw)

            self.v_cmd_hold = v_cmd
            self.w_cmd_hold = w_cmd

            self.v_cmd_prev = v_cmd
            self.w_cmd_prev = w_cmd

        commands = torch.zeros((self.num_envs, 4), device=self.device)
        commands[:, 0] = self.v_cmd_hold
        commands[:, 1] = 0.0
        commands[:, 2] = self.w_cmd_hold
        commands[:, 3] = 0.0
        return commands


def get_local_pose(env):
    root_pos = env.root_states[:, 0:3]
    local_pos = root_pos - env.env_origins

    local_x = local_pos[:, 0]
    local_y = local_pos[:, 1]

    quat = env.base_quat
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )

    return local_x, local_y, yaw


def perturb_initial_pose(env, y_range=0.20, yaw_range=0.30):
    root_states = env.root_states.clone()

    y_offsets = torch.empty(env.num_envs, device=env.device).uniform_(-y_range, y_range)
    yaw_offsets = torch.empty(env.num_envs, device=env.device).uniform_(-yaw_range, yaw_range)

    root_states[:, 0] = env.env_origins[:, 0]
    root_states[:, 1] = env.env_origins[:, 1] + y_offsets
    root_states[:, 2] = env.root_states[:, 2]

    q = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
        yaw_offsets
    )
    root_states[:, 3:7] = q
    root_states[:, 7:13] = 0.0

    env.root_states[:] = root_states
    env.gym.set_actor_root_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)


def draw_bezier_paths(env, follower, z=0.02):
    if env.viewer is None:
        return

    env.gym.clear_lines(env.viewer)

    E = follower.path_points.shape[0]
    N = follower.path_points.shape[1]

    for i in range(E):
        origin = env.env_origins[i].detach().cpu().numpy()
        pts = follower.path_points[i].detach().cpu().numpy()

        for k in range(N - 1):
            p0 = np.array([origin[0] + pts[k, 0],     origin[1] + pts[k, 1],     z], dtype=np.float32)
            p1 = np.array([origin[0] + pts[k + 1, 0], origin[1] + pts[k + 1, 1], z], dtype=np.float32)

            color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            env.gym.add_lines(
                env.viewer,
                env.envs[i],
                1,
                np.concatenate([p0, p1]).astype(np.float32),
                color
            )


def draw_anchor_points(env, follower, z=0.05):
    if env.viewer is None:
        return

    E, K, _ = follower.anchor_points.shape

    for i in range(E):
        origin = env.env_origins[i].detach().cpu().numpy()
        pts = follower.anchor_points[i].detach().cpu().numpy()

        for k in range(K):
            x = origin[0] + pts[k, 0]
            y = origin[1] + pts[k, 1]

            p1 = np.array([x - 0.04, y, z], dtype=np.float32)
            p2 = np.array([x + 0.04, y, z], dtype=np.float32)
            p3 = np.array([x, y - 0.04, z], dtype=np.float32)
            p4 = np.array([x, y + 0.04, z], dtype=np.float32)

            color = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p1, p2]).astype(np.float32), color)
            env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p3, p4]).astype(np.float32), color)


def draw_lookahead_targets(env, follower, z=0.04):
    if env.viewer is None:
        return

    for i in range(follower.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()
        x_t = origin[0] + float(follower.target_x[i].item())
        y_t = origin[1] + float(follower.target_y[i].item())

        p1 = np.array([x_t - 0.05, y_t, z], dtype=np.float32)
        p2 = np.array([x_t + 0.05, y_t, z], dtype=np.float32)
        p3 = np.array([x_t, y_t - 0.05, z], dtype=np.float32)
        p4 = np.array([x_t, y_t + 0.05, z], dtype=np.float32)

        color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p1, p2]).astype(np.float32), color)
        env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p3, p4]).astype(np.float32), color)


def draw_nearest_points(env, follower, z=0.035):
    if env.viewer is None:
        return

    for i in range(follower.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()
        x_n = origin[0] + float(follower.nearest_x[i].item())
        y_n = origin[1] + float(follower.nearest_y[i].item())

        p1 = np.array([x_n - 0.03, y_n, z], dtype=np.float32)
        p2 = np.array([x_n + 0.03, y_n, z], dtype=np.float32)
        p3 = np.array([x_n, y_n - 0.03, z], dtype=np.float32)
        p4 = np.array([x_n, y_n + 0.03, z], dtype=np.float32)

        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p1, p2]).astype(np.float32), color)
        env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p3, p4]).astype(np.float32), color)


def print_status(step_idx, env, commands, follower, every=50):
    if step_idx % every != 0:
        return

    local_x, local_y, yaw = get_local_pose(env)
    _, _, signed_ey, e_psi, alpha = follower._compute_path_geometry(local_x, local_y, yaw)

    recover0 = bool(follower.in_recovery[0].item())

    print(
        f"[step {step_idx:04d}] "
        f"ey0={signed_ey[0].item():.3f}, "
        f"epsi0={e_psi[0].item():.3f}, "
        f"alpha0={alpha[0].item():.3f}, "
        f"Ld0={follower.current_lookahead[0].item():.3f}, "
        f"cmd_vx0={commands[0, 0].item():.3f}, "
        f"cmd_wz0={commands[0, 2].item():.3f}, "
        f"env_vx0={env.commands[0, 0].item():.3f}, "
        f"env_wz0={env.commands[0, 2].item():.3f}, "
        f"x0={local_x[0].item():.3f}, "
        f"y0={local_y[0].item():.3f}, "
        f"yaw0={yaw[0].item():.3f}, "
        f"recover0={recover0}"
    )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "resampling_time"):
            env_cfg.commands.resampling_time = 999999.0
        if hasattr(env_cfg.commands, "heading_command"):
            env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    print("env.num_envs =", env.num_envs)
    print("env.dt       =", env.dt)
    print("100 steps    =", 100 * env.dt, "seconds")

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    perturb_initial_pose(env, y_range=0.20, yaw_range=0.30)
    obs = env.get_observations()

    follower = BezierPathFollower(
        num_envs=env.num_envs,
        device=env.device,
        env_dt=env.dt
    )

    # ===== path 설정 =====
    anchor_mode = "split_lr"
    follower.setup_anchor_paths(anchor_mode=anchor_mode)

    # ===== 일반 추종 설정 =====
    follower.vx_min = 0.10
    follower.vx_max = 1.5
    follower.wz_min = -0.9
    follower.wz_max = 0.9

    follower.hold_steps = 5
    follower.dv_max = 0.15
    follower.dw_max = 0.15

    follower.v_base = 0.85
    follower.k_alpha = 2.2
    follower.k_yaw_damp = 0.0
    follower.k_v_alpha = 0.9
    follower.k_v_ey = 0.20

    # ===== 동적 lookahead =====
    follower.lookahead_base = 0.30
    follower.lookahead_kv = 0.20
    follower.lookahead_ka = 0.20
    follower.lookahead_min = 0.25
    follower.lookahead_max = 0.45
    follower.current_lookahead[:] = follower.lookahead_base

    # ===== recovery hysteresis =====
    follower.recover_enter_ey = 0.65
    follower.recover_enter_epsi = 0.35
    follower.recover_exit_ey = 0.48
    follower.recover_exit_epsi = 0.15
    follower.recover_vx = 0.10
    follower.recover_wz = 3.9

    print("bezier heuristic settings:")
    print("  anchor_mode         =", anchor_mode)
    print("  vx range            =", (follower.vx_min, follower.vx_max))
    print("  wz range            =", (follower.wz_min, follower.wz_max))
    print("  hold_steps          =", follower.hold_steps)
    print("  dv_max              =", follower.dv_max)
    print("  dw_max              =", follower.dw_max)
    print("  v_base              =", follower.v_base)
    print("  k_alpha             =", follower.k_alpha)
    print("  k_yaw_damp          =", follower.k_yaw_damp)
    print("  k_v_alpha           =", follower.k_v_alpha)
    print("  k_v_ey              =", follower.k_v_ey)

    print("dynamic lookahead settings:")
    print("  lookahead_base      =", follower.lookahead_base)
    print("  lookahead_kv        =", follower.lookahead_kv)
    print("  lookahead_ka        =", follower.lookahead_ka)
    print("  lookahead_min       =", follower.lookahead_min)
    print("  lookahead_max       =", follower.lookahead_max)

    print("recovery settings:")
    print("  recover_enter_ey    =", follower.recover_enter_ey)
    print("  recover_enter_epsi  =", follower.recover_enter_epsi)
    print("  recover_exit_ey     =", follower.recover_exit_ey)
    print("  recover_exit_epsi   =", follower.recover_exit_epsi)
    print("  recover_vx          =", follower.recover_vx)
    print("  recover_wz          =", follower.recover_wz)

    total_steps = 10 * int(env.max_episode_length)

    for i in range(total_steps):
        local_x, local_y, yaw = get_local_pose(env)

        # geometry 업데이트
        follower._compute_path_geometry(local_x, local_y, yaw)

        draw_bezier_paths(env, follower, z=0.02)
        draw_anchor_points(env, follower, z=0.05)
        draw_nearest_points(env, follower, z=0.035)
        draw_lookahead_targets(env, follower, z=0.04)

        commands = follower.compute_commands(i, local_x, local_y, yaw)
        env.commands[:, :] = commands

        if i % 50 == 0:
            print("AFTER_SET", env.commands[0].detach().cpu().numpy())

        obs = env.get_observations()

        lower_actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(lower_actions.detach())

        if i % 50 == 0:
            print("AFTER_STEP", env.commands[0].detach().cpu().numpy())

        if torch.any(dones):
            done_ids = torch.nonzero(dones).flatten()
            if len(done_ids) > 0:
                root_states = env.root_states.clone()

                y_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.20, 0.20)
                yaw_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.30, 0.30)

                root_states[done_ids, 0] = env.env_origins[done_ids, 0]
                root_states[done_ids, 1] = env.env_origins[done_ids, 1] + y_offsets

                q = quat_from_euler_xyz(
                    torch.zeros(len(done_ids), device=env.device),
                    torch.zeros(len(done_ids), device=env.device),
                    yaw_offsets
                )
                root_states[done_ids, 3:7] = q
                root_states[done_ids, 7:13] = 0.0

                env.root_states[done_ids] = root_states[done_ids]
                env.gym.set_actor_root_state_tensor(
                    env.sim,
                    gymtorch.unwrap_tensor(env.root_states)
                )
                env.gym.refresh_actor_root_state_tensor(env.sim)

                follower.v_cmd_prev[done_ids] = 0.0
                follower.w_cmd_prev[done_ids] = 0.0
                follower.v_cmd_hold[done_ids] = 0.0
                follower.w_cmd_hold[done_ids] = 0.0
                follower.in_recovery[done_ids] = False
                follower.current_lookahead[done_ids] = follower.lookahead_base

        print_status(i, env, commands, follower, every=50)


if __name__ == "__main__":
    args = get_args()
    play(args)