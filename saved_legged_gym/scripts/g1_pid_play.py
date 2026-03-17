import isaacgym
import torch
import csv
from datetime import datetime
from pathlib import Path

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

from g1_pid_module import (
    DotsSplinePidFollower,
    arrange_envs_along_y,
    draw_command_arrows,
    draw_camera_debug,
    draw_path_and_dashes,
    draw_tracking_points,
    get_local_pose_rpy,
    perturb_initial_pose,
    print_status,
    reset_done_envs,
)

# RL 안 쓰고 PD로만 움직이는 코드임
# g1_pid_module.py에 PD 제어 모듈과 시각화 도구들이 구현되어 있음
# 이 코드는 실행만 맡음


class CsvSignalLogger:
    def __init__(self):
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = logs_dir / f"g1_pid_profile_{stamp}.csv"
        self.f = self.path.open("w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(
            [
                "step_idx",
                "time_sec",
                "mode_recovery",
                "n_visible",
                "lookahead_alpha",
                "u_err_near",
                "u_err_lookahead",
                "u_err_ctrl",
                "slope_scaled",
                "cmd_out_vx",
                "cmd_out_wz",
                "cmd_env_vx",
                "cmd_env_wz",
                "cmd_start_vx",
                "cmd_start_wz",
                "cmd_goal_vx",
                "cmd_goal_wz",
                "interp_countdown",
                "interp_ratio",
            ]
        )
        self._rows = 0
        print(f"[CSV] logging to: {self.path}")

    def log(self, step_idx, env, commands, follower):
        recover0 = bool(follower.in_recovery[0].item())
        nvis0 = float(follower.n_visible[0].item())
        use_la0 = bool(getattr(follower, "use_lookahead", False))
        alpha_normal = float(getattr(follower, "lookahead_alpha_normal", 0.0))
        alpha_recovery = float(getattr(follower, "lookahead_alpha_recovery", 0.0))
        alpha0 = alpha_recovery if recover0 else alpha_normal
        if (not use_la0) or (nvis0 < 2.0):
            alpha0 = 0.0

        u_near = float(follower.vision_u_err[0].item())
        u_la = float(follower.vision_u_err_la[0].item())
        u_ctrl = (1.0 - alpha0) * u_near + alpha0 * u_la

        interp_steps = max(1, int(follower.command_interp_steps))
        countdown = float(follower.interp_countdown[0].item())
        if interp_steps <= 1:
            interp_ratio = 1.0
        else:
            interp_ratio = float(max(0.0, min(1.0, (interp_steps - countdown) / float(interp_steps))))

        self.w.writerow(
            [
                int(step_idx),
                float(step_idx) * float(env.dt),
                1 if recover0 else 0,
                nvis0,
                alpha0,
                u_near,
                u_la,
                u_ctrl,
                float(follower.vision_slope[0].item()),
                float(commands[0, 0].item()),
                float(commands[0, 2].item()),
                float(env.commands[0, 0].item()),
                float(env.commands[0, 2].item()),
                float(follower.v_cmd_start[0].item()),
                float(follower.w_cmd_start[0].item()),
                float(follower.v_cmd_goal[0].item()),
                float(follower.w_cmd_goal[0].item()),
                countdown,
                interp_ratio,
            ]
        )
        self._rows += 1
        if self._rows % 100 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True  # 센서 노이즈
    env_cfg.domain_rand.randomize_friction = True  # 마찰계수 랜덤
    env_cfg.domain_rand.push_robots = False  # 외란
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 60.0  # 에피소드 길이

    # Disable internal random command scheduler.
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
        seed=None,  # 랜덤 시드
    )
    follower.setup_random_dotted_spline_paths()
    follower.print_config()
    csv_logger = CsvSignalLogger()

    total_steps = 10 * int(env.max_episode_length)

    try:
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

            commands = follower.compute_upper_command_from_vision(i)
            env.commands[:, :] = commands
            csv_logger.log(i, env, commands, follower)

            draw_path_and_dashes(env, follower)
            draw_tracking_points(env, follower)
            draw_camera_debug(env, follower, z_ground=follower.path_z)
            draw_command_arrows(env, follower, commands)

            # Important: refresh obs after injecting upper commands
            obs = env.get_observations()

            lower_actions = policy(obs.detach())
            obs, _, _rews, dones, _infos = env.step(lower_actions.detach())

            if torch.any(dones):
                done_ids = torch.nonzero(dones).flatten()
                reset_done_envs(env, done_ids, follower)

            print_status(i, env, commands, follower, every=50)
    finally:
        csv_logger.close()


if __name__ == "__main__":
    args = get_args()
    play(args)
