import isaacgym

import math
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class HeuristicCurveFollower:
    """
    원호(centerline arc) 기반 lookahead 휴리스틱.

    각 env마다:
    - 좌회전 원호 또는 우회전 원호를 하나 배정
    - 현재 위치에서 원호의 최근접점(nearest point)을 구하고
    - 그 지점에서 lookahead arc length만큼 앞의 target point를 생성
    - target point를 향하도록 (vx, 0, wz) 생성

    recovery mode + hysteresis 포함
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

        # ===== normal mode =====
        self.v_base = 1.0
        self.lookahead_dist = 0.45
        self.k_alpha = 2.2
        self.k_yaw_damp = 0.0
        self.k_v_alpha = 0.6
        self.k_v_ey = 0.15

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

        # ===== 곡선 파라미터 =====
        # env별로 center / radius / direction 저장
        self.radius = torch.full((self.num_envs,), 3.0, device=self.device)
        self.center_x = torch.zeros(self.num_envs, device=self.device)
        self.center_y = torch.zeros(self.num_envs, device=self.device)

        # +1: 좌회전(CCW), -1: 우회전(CW)
        self.turn_dir = torch.ones(self.num_envs, device=self.device)

        # 현재 lookahead target 저장(시각화용)
        self.target_x = torch.zeros(self.num_envs, device=self.device)
        self.target_y = torch.zeros(self.num_envs, device=self.device)

    def setup_arc_paths(self, mode="split_lr", radius=3.0):
        """
        각 env의 원호 파라미터 설정.

        local frame 기준:
        - 좌회전 원호: center = (0, +R), 시작점 (0,0), 초기 접선은 +x
        - 우회전 원호: center = (0, -R), 시작점 (0,0), 초기 접선은 +x
        """
        self.radius[:] = radius

        if mode == "left_only":
            self.turn_dir[:] = 1.0
        elif mode == "right_only":
            self.turn_dir[:] = -1.0
        elif mode == "split_lr":
            half = self.num_envs // 2
            self.turn_dir[:half] = 1.0
            self.turn_dir[half:] = -1.0
        else:
            raise ValueError(f"Unknown arc mode: {mode}")

        # center_x는 모두 0
        self.center_x[:] = 0.0

        # 좌회전이면 +R, 우회전이면 -R
        self.center_y = self.turn_dir * self.radius

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

    def _compute_arc_geometry(self, local_x, local_y, yaw):
        """
        현재 위치에서:
        - 원호 최근접점
        - 원호 lookahead target
        - signed radial error e_y
        - tangent heading error e_psi
        - target heading error alpha
        계산
        """
        dx = local_x - self.center_x
        dy = local_y - self.center_y

        r_now = torch.sqrt(dx * dx + dy * dy + 1e-8)

        # 현재 위치 방향 각도 (원 중심 기준)
        theta_now = torch.atan2(dy, dx)

        # 최근접점: 같은 theta에서 반지름만 R로 맞춘 점
        x_near = self.center_x + self.radius * torch.cos(theta_now)
        y_near = self.center_y + self.radius * torch.sin(theta_now)

        # signed radial error
        # 바깥이면 +, 안쪽이면 -
        e_y = r_now - self.radius

        # 최근접점에서의 경로 접선 각도
        # 좌회전(CCW): tangent = theta + pi/2
        # 우회전(CW):  tangent = theta - pi/2
        tangent_heading = theta_now + self.turn_dir * (math.pi / 2.0)
        tangent_heading = torch.atan2(torch.sin(tangent_heading), torch.cos(tangent_heading))

        e_psi = tangent_heading - yaw
        e_psi = torch.atan2(torch.sin(e_psi), torch.cos(e_psi))

        # lookahead target
        delta_theta = self.lookahead_dist / self.radius
        theta_target = theta_now + self.turn_dir * delta_theta

        x_t = self.center_x + self.radius * torch.cos(theta_target)
        y_t = self.center_y + self.radius * torch.sin(theta_target)

        # target heading error alpha
        target_heading = torch.atan2(y_t - local_y, x_t - local_x)
        alpha = target_heading - yaw
        alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))

        # 저장 (시각화용)
        self.target_x = x_t
        self.target_y = y_t

        return x_near, y_near, x_t, y_t, e_y, e_psi, alpha

    def compute_commands(self, step_idx, local_x, local_y, yaw):
        if step_idx % self.hold_steps == 0:
            _, _, _, _, e_y, e_psi, alpha = self._compute_arc_geometry(local_x, local_y, yaw)

            self._update_recovery_state(e_y, e_psi)

            # normal mode
            w_raw = self.k_alpha * alpha - self.k_yaw_damp * yaw
            v_raw = (
                self.v_base
                - self.k_v_alpha * torch.abs(alpha)
                - self.k_v_ey * torch.abs(e_y)
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
    """
    env origin 기준 local pose
    """
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


def draw_arc_centerlines(env, follower, arc_length=12.0, num_segments=60, z=0.02):
    """
    각 env의 원호 경로를 빨간 선으로 그림
    """
    if env.viewer is None:
        return

    env.gym.clear_lines(env.viewer)

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()

        R = float(follower.radius[i].item())
        cx_local = float(follower.center_x[i].item())
        cy_local = float(follower.center_y[i].item())
        turn_dir = float(follower.turn_dir[i].item())

        # 시작점 (0,0)을 지나는 theta_start
        theta_start = -turn_dir * (math.pi / 2.0)

        total_dtheta = arc_length / R

        thetas = []
        for k in range(num_segments + 1):
            ratio = k / num_segments
            theta = theta_start + turn_dir * total_dtheta * ratio
            thetas.append(theta)

        for k in range(num_segments):
            th0 = thetas[k]
            th1 = thetas[k + 1]

            x0_local = cx_local + R * math.cos(th0)
            y0_local = cy_local + R * math.sin(th0)

            x1_local = cx_local + R * math.cos(th1)
            y1_local = cy_local + R * math.sin(th1)

            p0 = np.array([origin[0] + x0_local, origin[1] + y0_local, z], dtype=np.float32)
            p1 = np.array([origin[0] + x1_local, origin[1] + y1_local, z], dtype=np.float32)

            color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            env.gym.add_lines(
                env.viewer,
                env.envs[i],
                1,
                np.concatenate([p0, p1]).astype(np.float32),
                color
            )


def draw_lookahead_targets(env, follower, z=0.04):
    """
    lookahead target을 초록 십자로 그림
    """
    if env.viewer is None:
        return

    for i in range(env.num_envs):
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


def print_status(step_idx, env, commands, follower, every=50):
    if step_idx % every != 0:
        return

    local_x, local_y, yaw = get_local_pose(env)
    _, _, _, _, e_y, e_psi, alpha = follower._compute_arc_geometry(local_x, local_y, yaw)

    recover0 = bool(follower.in_recovery[0].item())
    turn0 = "L" if follower.turn_dir[0].item() > 0 else "R"

    print(
        f"[step {step_idx:04d}] "
        f"turn0={turn0}, "
        f"ey0={e_y[0].item():.3f}, "
        f"epsi0={e_psi[0].item():.3f}, "
        f"alpha0={alpha[0].item():.3f}, "
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

    # 중요: 내부 command 개입 막기
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

    follower = HeuristicCurveFollower(
        num_envs=env.num_envs,
        device=env.device,
        env_dt=env.dt
    )

    # ===== arc 설정 =====
    # "left_only", "right_only", "split_lr"
    arc_mode = "split_lr"
    arc_radius = 3.0
    follower.setup_arc_paths(mode=arc_mode, radius=arc_radius)

    # ===== 일반 추종 설정 =====
    follower.vx_min = 0.10
    follower.vx_max = 1.5
    follower.wz_min = -0.9
    follower.wz_max = 0.9

    follower.hold_steps = 5
    follower.dv_max = 0.15
    follower.dw_max = 0.15

    follower.v_base = 1.0
    follower.lookahead_dist = 0.45
    follower.k_alpha = 2.2
    follower.k_yaw_damp = 0.0

    follower.k_v_alpha = 0.6
    follower.k_v_ey = 0.15

    # ===== recovery hysteresis =====
    follower.recover_enter_ey = 0.35
    follower.recover_enter_epsi = 0.35
    follower.recover_exit_ey = 0.18
    follower.recover_exit_epsi = 0.15
    follower.recover_vx = 0.10
    follower.recover_wz = 0.9

    print("curve heuristic settings:")
    print("  arc_mode            =", arc_mode)
    print("  arc_radius          =", arc_radius)
    print("  vx range            =", (follower.vx_min, follower.vx_max))
    print("  wz range            =", (follower.wz_min, follower.wz_max))
    print("  hold_steps          =", follower.hold_steps)
    print("  dv_max              =", follower.dv_max)
    print("  dw_max              =", follower.dw_max)
    print("  v_base              =", follower.v_base)
    print("  lookahead_dist      =", follower.lookahead_dist)
    print("  k_alpha             =", follower.k_alpha)
    print("  k_yaw_damp          =", follower.k_yaw_damp)
    print("  k_v_alpha           =", follower.k_v_alpha)
    print("  k_v_ey              =", follower.k_v_ey)

    print("recovery settings:")
    print("  recover_enter_ey    =", follower.recover_enter_ey)
    print("  recover_enter_epsi  =", follower.recover_enter_epsi)
    print("  recover_exit_ey     =", follower.recover_exit_ey)
    print("  recover_exit_epsi   =", follower.recover_exit_epsi)
    print("  recover_vx          =", follower.recover_vx)
    print("  recover_wz          =", follower.recover_wz)

    use_push = False
    push_interval_steps = 150
    push_max_y = 0.18

    print("push settings:")
    print("  use_push            =", use_push)
    print("  push_interval_steps =", push_interval_steps)
    print("  push_max_y          =", push_max_y)

    total_steps = 10 * int(env.max_episode_length)

    for i in range(total_steps):
        local_x, local_y, yaw = get_local_pose(env)

        # geometry update for visualization/logging
        follower._compute_arc_geometry(local_x, local_y, yaw)

        draw_arc_centerlines(env, follower, arc_length=12.0, num_segments=60, z=0.02)
        draw_lookahead_targets(env, follower, z=0.04)

        commands = follower.compute_commands(i, local_x, local_y, yaw)
        env.commands[:, :] = commands

        if i % 50 == 0:
            print("AFTER_SET", env.commands[0].detach().cpu().numpy())

        # 방금 넣은 command가 obs에 반영되도록 다시 계산
        obs = env.get_observations()

        lower_actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(lower_actions.detach())

        if i % 50 == 0:
            print("AFTER_STEP", env.commands[0].detach().cpu().numpy())

        if use_push:
            # 필요 시 나중에 활성화
            pass

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

        print_status(i, env, commands, follower, every=50)


if __name__ == "__main__":
    args = get_args()
    play(args)