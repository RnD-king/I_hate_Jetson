import isaacgym

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class HeuristicLineFollower:
    """
    직선 centerline(y=0)을 기준으로
    lookahead target 기반으로 (vx, 0, wz)를 생성하는 휴리스틱 상위 제어기.
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

        # ===== 갱신 주기 =====
        self.hold_steps = 5

        # ===== rate limit =====
        self.dv_max = 0.15
        self.dw_max = 0.15

        # ===== nominal speed / steering =====
        self.v_base = 1.0
        self.lookahead_dist = 0.45
        self.k_alpha = 2.2
        self.k_yaw_damp = 0.0

        # ===== speed shaping =====
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

    def _clip_and_rate_limit(self, v_raw, w_raw):
        v = torch.clamp(v_raw, self.vx_min, self.vx_max)
        w = torch.clamp(w_raw, self.wz_min, self.wz_max)

        v = torch.clamp(v, self.v_cmd_prev - self.dv_max, self.v_cmd_prev + self.dv_max)
        w = torch.clamp(w, self.w_cmd_prev - self.dw_max, self.w_cmd_prev + self.dw_max)

        return v, w

    def _update_recovery_state(self, y, psi):
        enter_mask = (torch.abs(y) > self.recover_enter_ey) | (torch.abs(psi) > self.recover_enter_epsi)
        exit_mask = (torch.abs(y) < self.recover_exit_ey) & (torch.abs(psi) < self.recover_exit_epsi)

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

    def compute_commands(self, step_idx, local_x, local_y, yaw):
        if step_idx % self.hold_steps == 0:
            x = local_x
            y = local_y
            psi = yaw

            self._update_recovery_state(y, psi)

            # line 위의 lookahead target
            x_t = x + self.lookahead_dist
            y_t = torch.zeros_like(y)

            dx = x_t - x
            dy = y_t - y

            target_heading = torch.atan2(dy, dx)

            alpha = target_heading - psi
            alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))

            # normal mode
            w_raw = self.k_alpha * alpha - self.k_yaw_damp * psi
            v_raw = (
                self.v_base
                - self.k_v_alpha * torch.abs(alpha)
                - self.k_v_ey * torch.abs(y)
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


def get_local_pose_errors_straight(env):
    """
    직선 centerline:
        local y = 0
        local heading = 0
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

    e_y = local_y
    e_psi = yaw

    return local_x, local_y, yaw, e_y, e_psi


def perturb_initial_pose(env, y_range=0.25, yaw_range=0.35):
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


def draw_centerlines(env, line_length=12.0):
    if env.viewer is None:
        return

    env.gym.clear_lines(env.viewer)

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()

        x0 = float(origin[0])
        y0 = float(origin[1])
        z0 = 0.02

        start = np.array([x0, y0, z0], dtype=np.float32)
        end = np.array([x0 + line_length, y0, z0], dtype=np.float32)

        vertices = np.concatenate([start, end]).astype(np.float32)
        color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        env.gym.add_lines(
            env.viewer,
            env.envs[i],
            1,
            vertices,
            color
        )


def draw_lookahead_targets(env, local_x, follower, z=0.04):
    if env.viewer is None:
        return

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()
        lx = float(local_x[i].item())
        x_t = origin[0] + lx + follower.lookahead_dist
        y_t = origin[1]

        p1 = np.array([x_t - 0.05, y_t, z], dtype=np.float32)
        p2 = np.array([x_t + 0.05, y_t, z], dtype=np.float32)

        p3 = np.array([x_t, y_t - 0.05, z], dtype=np.float32)
        p4 = np.array([x_t, y_t + 0.05, z], dtype=np.float32)

        color = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p1, p2]).astype(np.float32), color)
        env.gym.add_lines(env.viewer, env.envs[i], 1, np.concatenate([p3, p4]).astype(np.float32), color)


def apply_lateral_push(env, step_idx, interval_steps=150, max_push_y=0.12):
    if step_idx == 0 or step_idx % interval_steps != 0:
        return

    root_states = env.root_states.clone()
    push_y = torch.empty(env.num_envs, device=env.device).uniform_(-max_push_y, max_push_y)
    root_states[:, 8] += push_y

    env.root_states[:] = root_states
    env.gym.set_actor_root_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)

    print(f"[push @ step {step_idx}] push_y={push_y.detach().cpu().numpy()}")


def print_status(step_idx, env, commands, follower, every=50):
    if step_idx % every != 0:
        return

    local_x, local_y, yaw, e_y, e_psi = get_local_pose_errors_straight(env)

    dx = torch.full_like(local_y, follower.lookahead_dist)
    dy = -local_y
    target_heading = torch.atan2(dy, dx)
    alpha = target_heading - yaw
    alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))

    recover0 = bool(follower.in_recovery[0].item())

    print(
        f"[step {step_idx:04d}] "
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

    # ===== 중요: cmd_eval에서 했던 override 여기에도 넣기 =====
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

    perturb_initial_pose(env, y_range=0.25, yaw_range=0.35)
    obs = env.get_observations()

    follower = HeuristicLineFollower(
        num_envs=env.num_envs,
        device=env.device,
        env_dt=env.dt
    )

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

    # ===== recovery hysteresis 설정 =====
    follower.recover_enter_ey = 0.35
    follower.recover_enter_epsi = 0.35
    follower.recover_exit_ey = 0.18
    follower.recover_exit_epsi = 0.15
    follower.recover_vx = 0.10
    follower.recover_wz = 0.9

    print("lookahead heuristic settings:")
    print("  vx range           =", (follower.vx_min, follower.vx_max))
    print("  wz range           =", (follower.wz_min, follower.wz_max))
    print("  hold_steps         =", follower.hold_steps)
    print("  dv_max             =", follower.dv_max)
    print("  dw_max             =", follower.dw_max)
    print("  v_base             =", follower.v_base)
    print("  lookahead_dist     =", follower.lookahead_dist)
    print("  k_alpha            =", follower.k_alpha)
    print("  k_yaw_damp         =", follower.k_yaw_damp)
    print("  k_v_alpha          =", follower.k_v_alpha)
    print("  k_v_ey             =", follower.k_v_ey)

    print("recovery hysteresis settings:")
    print("  recover_enter_ey   =", follower.recover_enter_ey)
    print("  recover_enter_epsi =", follower.recover_enter_epsi)
    print("  recover_exit_ey    =", follower.recover_exit_ey)
    print("  recover_exit_epsi  =", follower.recover_exit_epsi)
    print("  recover_vx         =", follower.recover_vx)
    print("  recover_wz         =", follower.recover_wz)

    use_push = False
    push_interval_steps = 150
    push_max_y = 0.18

    print("push settings:")
    print("  use_push            =", use_push)
    print("  push_interval_steps =", push_interval_steps)
    print("  push_max_y          =", push_max_y)

    total_steps = 10 * int(env.max_episode_length)

    for i in range(total_steps):
        local_x, local_y, yaw, e_y, e_psi = get_local_pose_errors_straight(env)

        draw_centerlines(env, line_length=12.0)
        draw_lookahead_targets(env, local_x, follower, z=0.04)

        commands = follower.compute_commands(i, local_x, local_y, yaw)

        env.commands[:, :] = commands

        if i % 50 == 0:
            print("AFTER_SET", env.commands[0].detach().cpu().numpy())

        # 중요: 방금 넣은 command가 observation에 반영되도록 다시 계산
        obs = env.get_observations()

        lower_actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(lower_actions.detach())

        if i % 50 == 0:
            print("AFTER_STEP", env.commands[0].detach().cpu().numpy())

        if use_push:
            apply_lateral_push(
                env,
                i,
                interval_steps=push_interval_steps,
                max_push_y=push_max_y
            )

        if torch.any(dones):
            done_ids = torch.nonzero(dones).flatten()
            if len(done_ids) > 0:
                root_states = env.root_states.clone()

                y_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.25, 0.25)
                yaw_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.35, 0.35)

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