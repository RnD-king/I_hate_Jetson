import isaacgym

import math
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_to_rpy(quat: torch.Tensor):
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_local_pose_rpy(env):
    root_pos = env.root_states[:, 0:3]
    local_pos = root_pos - env.env_origins
    local_x = local_pos[:, 0]
    local_y = local_pos[:, 1]

    roll, pitch, yaw = quat_to_rpy(env.base_quat)
    return local_x, local_y, roll, pitch, yaw


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
        yaw_offsets,
    )
    root_states[:, 3:7] = q
    root_states[:, 7:13] = 0.0

    env.root_states[:] = root_states
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)


def reset_done_envs(env, done_ids, follower):
    if len(done_ids) == 0:
        return

    root_states = env.root_states.clone()
    y_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.22, 0.22)
    yaw_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.35, 0.35)

    root_states[done_ids, 0] = env.env_origins[done_ids, 0]
    root_states[done_ids, 1] = env.env_origins[done_ids, 1] + y_offsets

    q = quat_from_euler_xyz(
        torch.zeros(len(done_ids), device=env.device),
        torch.zeros(len(done_ids), device=env.device),
        yaw_offsets,
    )
    root_states[done_ids, 3:7] = q
    root_states[done_ids, 7:13] = 0.0

    env.root_states[done_ids] = root_states[done_ids]
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)

    follower.reset_env_state(done_ids)


def arrange_envs_along_y(env, y_gap=2.6):
    if env.num_envs != 2:
        return

    # Force two env origins to be separated along +Y/-Y for clearer visualization.
    env.env_origins[0, 0] = 0.0
    env.env_origins[1, 0] = 0.0
    env.env_origins[0, 1] = -0.5 * y_gap
    env.env_origins[1, 1] = +0.5 * y_gap
    env.env_origins[:, 2] = 0.0

    root_states = env.root_states.clone()
    root_states[:, 0] = env.env_origins[:, 0]
    root_states[:, 1] = env.env_origins[:, 1]
    root_states[:, 7:13] = 0.0

    env.root_states[:] = root_states
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)


def rot_x(a):
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(a):
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(a):
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rpy_to_rot(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


@dataclass
class CameraCfg:
    width: int = 640
    height: int = 480
    fx: float = 360.0
    fy: float = 360.0
    cx: float = 320.0
    cy: float = 240.0
    t_base_cam: np.ndarray = field(default_factory=lambda: np.array([0.18, 0.0, 0.38], dtype=np.float32))
    mount_rpy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.60, 0.0], dtype=np.float32))
    min_depth_x: float = 0.15
    max_depth_x: float = 4.0


class DotsSplinePidFollower:
    """
    One-file heuristic pipeline:
    - random dotted points on floor (dash samples from spline centerline)
    - virtual camera projection (no IMU roll/pitch correction)
    - vision-only upper controller outputting (vx, wz)
    """

    def __init__(self, num_envs, device, env_dt, seed=7):
        self.num_envs = num_envs
        self.device = device
        self.env_dt = env_dt
        self.rng = np.random.default_rng(seed)

        # ----- command envelope -----
        self.vx_min = 0.10
        self.vx_max = 1.20
        self.wz_min = -0.90
        self.wz_max = 0.90

        self.hold_steps = 3
        self.dv_max = 0.12
        self.dw_max = 0.20

        # ----- vision-only controller -----
        self.v_base = 0.85
        self.k_u = 1.40
        self.k_slope = 0.90
        self.k_v_u = 0.35
        self.k_v_slope = 0.25

        # ----- recovery (vision only) -----
        self.recover_enter_nvis = 1.0
        self.recover_exit_nvis = 3.0
        self.recover_enter_u = 0.70
        self.recover_exit_u = 0.35
        self.recover_vx = 0.12
        self.recover_wz = 0.75

        # ----- camera/perception -----
        self.cam_cfg = CameraCfg()
        self.use_rp_stabilization = False
        self.imu_noise_std_deg = 0.0

        self.max_centers = 8
        self.per_pt_dropout_prob = 0.12
        self.burst_dropout_prob = 0.04
        self.pixel_jitter_std = 1.2

        # ----- path -----
        self.num_waypoints = 26
        self.samples_per_seg = 36
        self.path_draw_stride = 6
        self.path_z = 0.02
        self.dash_len = 0.28
        self.gap_len = 0.22

        self.waypoints = None         # [E, K, 2]
        self.path_points = None       # [E, N, 2]
        self.path_s = None            # [E, N]
        self.path_heading = None      # [E, N]
        self.path_curvature = None    # [E, N]
        self.dash_points = []         # list[tensor(Mi,2)]

        # ----- runtime state -----
        self.in_recovery = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.v_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self.v_cmd_hold = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_hold = torch.zeros(self.num_envs, device=self.device)

        self.nearest_xy = torch.zeros((self.num_envs, 2), device=self.device)  # kept for optional debug
        self.target_xy = torch.zeros((self.num_envs, 2), device=self.device)   # kept for optional debug

        self.centers_uv = torch.zeros((self.num_envs, self.max_centers, 2), device=self.device)
        self.centers_valid = torch.zeros((self.num_envs, self.max_centers), device=self.device, dtype=torch.bool)
        self.vision_u_err = torch.zeros(self.num_envs, device=self.device)
        self.vision_slope = torch.zeros(self.num_envs, device=self.device)
        self.n_visible = torch.zeros(self.num_envs, device=self.device)
        self.visible_local_points = [torch.zeros((0, 2), device=self.device) for _ in range(self.num_envs)]
        self.debug_cam_pos_w = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.debug_cam_r_wc = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], self.num_envs, axis=0)

    # -------------------------------------------------------------------------
    # Path generation
    # -------------------------------------------------------------------------
    def _catmull_rom_chain(self, pts_xy: np.ndarray, samples_per_seg: int):
        out = []
        k = pts_xy.shape[0]
        for i in range(k - 1):
            p0 = pts_xy[max(i - 1, 0)]
            p1 = pts_xy[i]
            p2 = pts_xy[i + 1]
            p3 = pts_xy[min(i + 2, k - 1)]

            t_values = np.linspace(0.0, 1.0, samples_per_seg, dtype=np.float32)
            for t in t_values:
                t2 = t * t
                t3 = t2 * t
                c = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                out.append(c)
            if i > 0:
                out.pop(-samples_per_seg)
        return np.asarray(out, dtype=np.float32)

    def _compute_heading_curvature(self, path_xy: torch.Tensor, s: torch.Tensor):
        n = path_xy.shape[0]
        prev_idx = torch.clamp(torch.arange(n, device=self.device) - 1, 0, n - 1)
        next_idx = torch.clamp(torch.arange(n, device=self.device) + 1, 0, n - 1)

        tangent = path_xy[next_idx] - path_xy[prev_idx]
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])

        d_heading = wrap_to_pi_torch(heading[next_idx] - heading[prev_idx])
        ds = s[next_idx] - s[prev_idx]
        ds = torch.clamp(ds, min=1e-4)
        curvature = d_heading / ds
        return heading, curvature

    def _make_waypoints(self, env_idx: int):
        x = np.linspace(0.0, 34.0, self.num_waypoints, dtype=np.float32)
        y = np.zeros_like(x)

        sign = 1.0 if env_idx < (self.num_envs // 2) else -1.0
        rand_steps = self.rng.uniform(-0.34, 0.34, size=self.num_waypoints - 1)

        for i in range(1, self.num_waypoints):
            wave = 0.12 * sign * math.sin(0.44 * i)
            y[i] = np.clip(y[i - 1] + rand_steps[i - 1] + wave, -2.1, 2.1)

        y[0] = 0.0
        y[-1] *= 0.55
        return np.stack([x, y], axis=1)

    def _build_dash_points(self, path_xy: torch.Tensor, path_s: torch.Tensor):
        cycle = self.dash_len + self.gap_len
        phase = torch.remainder(path_s, cycle)
        mask = phase < self.dash_len
        pts = path_xy[mask]
        return pts

    def setup_random_dotted_spline_paths(self):
        all_waypoints = []
        all_paths = []
        all_s = []
        all_heading = []
        all_curvature = []
        self.dash_points = []

        for env_idx in range(self.num_envs):
            wp_np = self._make_waypoints(env_idx)
            dense_np = self._catmull_rom_chain(wp_np, self.samples_per_seg)

            path_xy = torch.tensor(dense_np, dtype=torch.float32, device=self.device)
            diffs = path_xy[1:] - path_xy[:-1]
            seg_len = torch.sqrt(torch.sum(diffs * diffs, dim=1) + 1e-8)
            s = torch.zeros(path_xy.shape[0], device=self.device)
            s[1:] = torch.cumsum(seg_len, dim=0)

            heading, curvature = self._compute_heading_curvature(path_xy, s)
            dashes = self._build_dash_points(path_xy, s)

            all_waypoints.append(torch.tensor(wp_np, dtype=torch.float32, device=self.device))
            all_paths.append(path_xy)
            all_s.append(s)
            all_heading.append(heading)
            all_curvature.append(curvature)
            self.dash_points.append(dashes)

        self.waypoints = torch.stack(all_waypoints, dim=0)
        self.path_points = torch.stack(all_paths, dim=0)
        self.path_s = torch.stack(all_s, dim=0)
        self.path_heading = torch.stack(all_heading, dim=0)
        self.path_curvature = torch.stack(all_curvature, dim=0)

    # -------------------------------------------------------------------------
    # Perception: world points -> camera -> projected centers (u,v)
    # -------------------------------------------------------------------------
    def _camera_pose_world(self, base_pos_w, roll, pitch, yaw):
        # No IMU-based roll/pitch compensation in this mode.
        r_wb = rpy_to_rot(roll, pitch, yaw)

        r_bc = rpy_to_rot(
            float(self.cam_cfg.mount_rpy[0]),
            float(self.cam_cfg.mount_rpy[1]),
            float(self.cam_cfg.mount_rpy[2]),
        )
        r_wc = r_wb @ r_bc
        cam_pos = base_pos_w + (r_wb @ self.cam_cfg.t_base_cam)
        return cam_pos, r_wc

    def _project_world_points(self, world_pts, cam_pos, r_wc):
        rel = world_pts - cam_pos[None, :]
        p_cam = rel @ r_wc  # cam frame: +x forward, +y left, +z up

        x = p_cam[:, 0]
        y = p_cam[:, 1]
        z = p_cam[:, 2]

        valid = (x > self.cam_cfg.min_depth_x) & (x < self.cam_cfg.max_depth_x)
        if np.count_nonzero(valid) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        x = x[valid]
        y = y[valid]
        z = z[valid]
        idx_valid = np.nonzero(valid)[0]

        u = self.cam_cfg.cx - self.cam_cfg.fx * (y / x)
        v = self.cam_cfg.cy - self.cam_cfg.fy * (z / x)
        uv = np.stack([u, v], axis=1)

        in_img = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < float(self.cam_cfg.width))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < float(self.cam_cfg.height))
        )
        uv = uv[in_img]
        idx_valid = idx_valid[in_img]
        return uv.astype(np.float32), idx_valid.astype(np.int64)

    def update_perception(self, local_x, local_y, base_z, roll, pitch, yaw, env_origins):
        self.centers_uv.zero_()
        self.centers_valid.zero_()
        self.vision_u_err.zero_()
        self.vision_slope.zero_()
        self.n_visible.zero_()

        for i in range(self.num_envs):
            origin = env_origins[i].detach().cpu().numpy()
            dash_local = self.dash_points[i].detach().cpu().numpy()
            dash_world = np.concatenate(
                [
                    origin[0:2][None, :] + dash_local,
                    np.full((dash_local.shape[0], 1), self.path_z, dtype=np.float32),
                ],
                axis=1,
            )

            base_pos_w = np.array(
                [
                    origin[0] + float(local_x[i].item()),
                    origin[1] + float(local_y[i].item()),
                    float(base_z[i].item()),
                ],
                dtype=np.float32,
            )

            cam_pos, r_wc = self._camera_pose_world(
                base_pos_w=base_pos_w,
                roll=float(roll[i].item()),
                pitch=float(pitch[i].item()),
                yaw=float(yaw[i].item()),
            )
            self.debug_cam_pos_w[i] = cam_pos
            self.debug_cam_r_wc[i] = r_wc

            uv, idx_map = self._project_world_points(dash_world, cam_pos, r_wc)
            if uv.shape[0] == 0:
                self.visible_local_points[i] = torch.zeros((0, 2), device=self.device)
                continue

            # per-point dropout
            keep_mask = self.rng.uniform(0.0, 1.0, size=uv.shape[0]) > self.per_pt_dropout_prob
            uv = uv[keep_mask]
            idx_map = idx_map[keep_mask]

            if uv.shape[0] == 0:
                self.visible_local_points[i] = torch.zeros((0, 2), device=self.device)
                continue

            # burst dropout
            if self.rng.uniform(0.0, 1.0) < self.burst_dropout_prob:
                pick = self.rng.integers(0, uv.shape[0])
                uv = uv[pick:pick + 1]
                idx_map = idx_map[pick:pick + 1]

            # pixel jitter
            uv += self.rng.normal(0.0, self.pixel_jitter_std, size=uv.shape).astype(np.float32)
            uv[:, 0] = np.clip(uv[:, 0], 0.0, float(self.cam_cfg.width - 1))
            uv[:, 1] = np.clip(uv[:, 1], 0.0, float(self.cam_cfg.height - 1))

            # sort by v(desc): bottom-most points first
            order = np.argsort(-uv[:, 1])
            uv = uv[order]
            idx_map = idx_map[order]

            keep_n = min(self.max_centers, uv.shape[0])
            uv = uv[:keep_n]
            idx_map = idx_map[:keep_n]

            self.centers_uv[i, :keep_n] = torch.tensor(uv, device=self.device)
            self.centers_valid[i, :keep_n] = True
            self.n_visible[i] = float(keep_n)

            local_points = self.dash_points[i][torch.tensor(idx_map, device=self.device, dtype=torch.long)]
            self.visible_local_points[i] = local_points

            # simple 2D vision features
            u_bottom = float(uv[0, 0])
            u_err = (u_bottom - self.cam_cfg.cx) / max(self.cam_cfg.cx, 1.0)
            self.vision_u_err[i] = u_err

            if keep_n >= 2:
                vv = uv[:, 1]
                uu = uv[:, 0]
                a, _b = np.polyfit(vv, uu, 1)
                self.vision_slope[i] = float(a) / 120.0
            else:
                self.vision_slope[i] = 0.0

    # -------------------------------------------------------------------------
    # Vision-only control
    # -------------------------------------------------------------------------
    def _update_recovery_state_vision(self):
        enter = (self.n_visible <= self.recover_enter_nvis) | (torch.abs(self.vision_u_err) > self.recover_enter_u)
        exit_ = (self.n_visible >= self.recover_exit_nvis) & (torch.abs(self.vision_u_err) < self.recover_exit_u)
        self.in_recovery = torch.where(enter, torch.ones_like(self.in_recovery), self.in_recovery)
        self.in_recovery = torch.where(exit_, torch.zeros_like(self.in_recovery), self.in_recovery)

    def _clip_and_rate_limit(self, v_raw, w_raw):
        v = torch.clamp(v_raw, self.vx_min, self.vx_max)
        w = torch.clamp(w_raw, self.wz_min, self.wz_max)
        v = torch.clamp(v, self.v_cmd_prev - self.dv_max, self.v_cmd_prev + self.dv_max)
        w = torch.clamp(w, self.w_cmd_prev - self.dw_max, self.w_cmd_prev + self.dw_max)
        return v, w

    def compute_commands(self, step_idx, local_x, local_y, yaw):
        if step_idx % self.hold_steps == 0:
            self._update_recovery_state_vision()

            w_nom = -self.k_u * self.vision_u_err - self.k_slope * self.vision_slope
            v_nom = (
                self.v_base
                - self.k_v_u * torch.abs(self.vision_u_err)
                - self.k_v_slope * torch.abs(self.vision_slope)
            )

            # If almost no perception, keep turning trend but slow down.
            no_vis = self.n_visible < 0.5
            low_vis = self.n_visible < 2.0
            w_nom = torch.where(low_vis, 0.90 * self.w_cmd_prev, w_nom)
            v_nom = torch.where(low_vis, torch.full_like(v_nom, 0.18), v_nom)

            # Recovery: vision-only hysteresis (no GT geometry).
            turn_sign = torch.sign(w_nom)
            turn_sign = torch.where(turn_sign == 0.0, torch.sign(self.w_cmd_prev), turn_sign)
            turn_sign = torch.where(turn_sign == 0.0, torch.ones_like(turn_sign), turn_sign)

            recover_v = torch.full_like(v_nom, self.recover_vx)
            recover_w = self.recover_wz * turn_sign
            v_raw = v_nom
            w_raw = w_nom
            v_raw = torch.where(self.in_recovery, recover_v, v_raw)
            w_raw = torch.where(self.in_recovery, recover_w, w_raw)

            # Fully blind frame: extremely conservative
            v_raw = torch.where(no_vis, torch.full_like(v_raw, 0.10), v_raw)
            w_raw = torch.where(no_vis, 0.95 * self.w_cmd_prev, w_raw)

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

    def reset_env_state(self, done_ids):
        self.in_recovery[done_ids] = False
        self.v_cmd_prev[done_ids] = 0.0
        self.w_cmd_prev[done_ids] = 0.0
        self.v_cmd_hold[done_ids] = 0.0
        self.w_cmd_hold[done_ids] = 0.0


def draw_path_and_dashes(env, follower: DotsSplinePidFollower):
    if env.viewer is None:
        return

    env.gym.clear_lines(env.viewer)

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()
        path = follower.path_points[i].detach().cpu().numpy()

        # centerline (red)
        for k in range(0, path.shape[0] - 1, follower.path_draw_stride):
            p0 = np.array([origin[0] + path[k, 0], origin[1] + path[k, 1], follower.path_z], dtype=np.float32)
            p1 = np.array([origin[0] + path[k + 1, 0], origin[1] + path[k + 1, 1], follower.path_z], dtype=np.float32)
            env.gym.add_lines(
                env.viewer,
                env.envs[i],
                1,
                np.concatenate([p0, p1]).astype(np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )

        # dash points (white small crosses)
        dashes = follower.dash_points[i].detach().cpu().numpy()
        for p in dashes[::6]:
            x = origin[0] + p[0]
            y = origin[1] + p[1]
            z = follower.path_z + 0.01
            _draw_cross_thick(
                env=env,
                env_handle=env.envs[i],
                x=x,
                y=y,
                z=z,
                half_len=0.02,
                color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                thickness=0.008,
            )


def _draw_cross_thick(env, env_handle, x, y, z, half_len, color, thickness=0.008):
    """
    Isaac Gym line width is fixed, so draw 3 parallel lines per axis
    to make crosses appear ~2x thicker.
    """
    offs = (0.0, +0.5 * thickness, -0.5 * thickness)

    for dy in offs:
        p1 = np.array([x - half_len, y + dy, z], dtype=np.float32)
        p2 = np.array([x + half_len, y + dy, z], dtype=np.float32)
        env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([p1, p2]).astype(np.float32), color)

    for dx in offs:
        p3 = np.array([x + dx, y - half_len, z], dtype=np.float32)
        p4 = np.array([x + dx, y + half_len, z], dtype=np.float32)
        env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([p3, p4]).astype(np.float32), color)


def draw_tracking_points(env, follower: DotsSplinePidFollower):
    if env.viewer is None:
        return

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()

        # visible projected points (cyan) mapped back to world dash points
        vis = follower.visible_local_points[i]
        if vis.numel() > 0:
            vis_np = vis.detach().cpu().numpy()
            for p in vis_np:
                xv = origin[0] + p[0]
                yv = origin[1] + p[1]
                zv = follower.path_z + 0.04
                _draw_cross_thick(
                    env=env,
                    env_handle=env.envs[i],
                    x=xv,
                    y=yv,
                    z=zv,
                    half_len=0.018,
                    color=np.array([0.0, 1.0, 1.0], dtype=np.float32),
                    thickness=0.008,
                )


def _pixel_ray_to_world_dir(u, v, cam_cfg: CameraCfg, r_wc):
    # Inverse of projection:
    # u = cx - fx*(y/x), v = cy - fy*(z/x), with x>0
    x = 1.0
    y = -((u - cam_cfg.cx) / cam_cfg.fx) * x
    z = -((v - cam_cfg.cy) / cam_cfg.fy) * x
    dir_cam = np.array([x, y, z], dtype=np.float32)
    dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-8)
    dir_world = dir_cam @ r_wc.T
    return dir_world


def _intersect_ground(cam_pos, dir_world, z_ground):
    dz = float(dir_world[2])
    if abs(dz) < 1e-6:
        return None
    t = (z_ground - float(cam_pos[2])) / dz
    if t <= 0.0:
        return None
    p = cam_pos + t * dir_world
    return p.astype(np.float32)


def _project_frame_to_ground(cam_pos, r_wc, cam_cfg: CameraCfg, z_ground):
    w = float(cam_cfg.width - 1)
    h = float(cam_cfg.height - 1)
    cy = float(cam_cfg.cy)

    corner_sets = [
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],                 # full frame
        [(0.0, cy), (w, cy), (w, h), (0.0, h)],                   # lower half fallback
        [(0.0, min(cy + 40.0, h)), (w, min(cy + 40.0, h)), (w, h), (0.0, h)],  # lower-biased fallback
    ]

    for corners_uv in corner_sets:
        ground_pts = []
        for (u, v) in corners_uv:
            d_w = _pixel_ray_to_world_dir(u, v, cam_cfg, r_wc)
            p = _intersect_ground(cam_pos, d_w, z_ground=z_ground)
            if p is None:
                ground_pts = []
                break
            ground_pts.append(p)
        if len(ground_pts) == 4:
            return ground_pts

    return []


def draw_camera_debug(env, follower: DotsSplinePidFollower, z_ground=0.02):
    if env.viewer is None:
        return

    for i in range(env.num_envs):
        cam_pos = follower.debug_cam_pos_w[i]
        r_wc = follower.debug_cam_r_wc[i]

        # Camera position marker (magenta cross)
        z = cam_pos[2]
        _draw_cross_thick(
            env=env,
            env_handle=env.envs[i],
            x=float(cam_pos[0]),
            y=float(cam_pos[1]),
            z=float(z),
            half_len=0.03,
            color=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            thickness=0.010,
        )

        # Frustum-to-ground footprint rectangle (with fallback if top rays miss ground).
        ground_pts = _project_frame_to_ground(
            cam_pos=cam_pos,
            r_wc=r_wc,
            cam_cfg=follower.cam_cfg,
            z_ground=z_ground,
        )

        if len(ground_pts) == 4:
            rect_col = np.array([1.0, 0.5, 0.0], dtype=np.float32)
            for k in range(4):
                p0 = ground_pts[k]
                p1 = ground_pts[(k + 1) % 4]
                env.gym.add_lines(
                    env.viewer,
                    env.envs[i],
                    1,
                    np.concatenate([p0, p1]).astype(np.float32),
                    rect_col,
                )

            # Draw 4 rays for visibility debugging.
            ray_col = np.array([1.0, 0.3, 0.3], dtype=np.float32)
            for p in ground_pts:
                env.gym.add_lines(
                    env.viewer,
                    env.envs[i],
                    1,
                    np.concatenate([cam_pos.astype(np.float32), p]).astype(np.float32),
                    ray_col,
                )


def print_status(step_idx, env, commands, follower: DotsSplinePidFollower, every=50):
    if step_idx % every != 0:
        return

    recover0 = bool(follower.in_recovery[0].item())
    nvis0 = int(follower.n_visible[0].item())
    uerr0 = float(follower.vision_u_err[0].item())
    slope0 = float(follower.vision_slope[0].item())

    print(
        f"[step {step_idx:04d}] "
        f"nvis0={nvis0}, "
        f"uerr0={uerr0:.3f}, "
        f"slope0={slope0:.3f}, "
        f"cmd_vx0={commands[0, 0].item():.3f}, "
        f"cmd_wz0={commands[0, 2].item():.3f}, "
        f"env_vx0={env.commands[0, 0].item():.3f}, "
        f"env_wz0={env.commands[0, 2].item():.3f}, "
        f"recover0={recover0}"
    )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 2
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # Disable internal random command scheduler to keep upper command deterministic.
    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "resampling_time"):
            env_cfg.commands.resampling_time = 999999.0
        if hasattr(env_cfg.commands, "heading_command"):
            env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    arrange_envs_along_y(env, y_gap=2.6)
    obs = env.get_observations()

    print("env.num_envs =", env.num_envs)
    print("env.dt       =", env.dt)
    print("100 steps    =", 100 * env.dt, "seconds")

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    perturb_initial_pose(env, y_range=0.06, yaw_range=0.30)
    obs = env.get_observations()

    follower = DotsSplinePidFollower(
        num_envs=env.num_envs,
        device=env.device,
        env_dt=env.dt,
        seed=11,
    )
    follower.setup_random_dotted_spline_paths()

    print("dots+spline+vision-only settings:")
    print("  vx range             =", (follower.vx_min, follower.vx_max))
    print("  wz range             =", (follower.wz_min, follower.wz_max))
    print("  hold_steps           =", follower.hold_steps)
    print("  dv_max               =", follower.dv_max)
    print("  dw_max               =", follower.dw_max)
    print("  v_base               =", follower.v_base)
    print("  k_u                  =", follower.k_u)
    print("  k_slope              =", follower.k_slope)
    print("  k_v_u                =", follower.k_v_u)
    print("  k_v_slope            =", follower.k_v_slope)
    print("  recover_enter_nvis   =", follower.recover_enter_nvis)
    print("  recover_exit_nvis    =", follower.recover_exit_nvis)
    print("  recover_enter_u      =", follower.recover_enter_u)
    print("  recover_exit_u       =", follower.recover_exit_u)
    print("  use_rp_stabilization =", follower.use_rp_stabilization)
    print("  per_pt_dropout_prob  =", follower.per_pt_dropout_prob)
    print("  burst_dropout_prob   =", follower.burst_dropout_prob)
    print("  pixel_jitter_std     =", follower.pixel_jitter_std)
    print("  max_centers          =", follower.max_centers)

    total_steps = 10 * int(env.max_episode_length)

    for i in range(total_steps):
        local_x, local_y, roll, pitch, yaw = get_local_pose_rpy(env)
        base_z = env.root_states[:, 2]

        follower.update_perception(
            local_x=local_x,
            local_y=local_y,
            base_z=base_z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            env_origins=env.env_origins,
        )

        commands = follower.compute_commands(i, local_x, local_y, yaw)
        env.commands[:, :] = commands

        draw_path_and_dashes(env, follower)
        draw_tracking_points(env, follower)
        draw_camera_debug(env, follower, z_ground=follower.path_z)

        if i % 50 == 0:
            print("AFTER_SET", env.commands[0].detach().cpu().numpy())

        # Recompute obs so the just-injected command is reflected in policy input.
        obs = env.get_observations()

        lower_actions = policy(obs.detach())
        obs, _, _rews, dones, _infos = env.step(lower_actions.detach())

        if i % 50 == 0:
            print("AFTER_STEP", env.commands[0].detach().cpu().numpy())

        if torch.any(dones):
            done_ids = torch.nonzero(dones).flatten()
            reset_done_envs(env, done_ids, follower)

        print_status(i, env, commands, follower, every=50)


if __name__ == "__main__":
    args = get_args()
    play(args)


# play_dots_spline_pid_follower.py 