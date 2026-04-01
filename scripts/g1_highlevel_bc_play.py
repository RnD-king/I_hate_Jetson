import argparse
import sys

import isaacgym
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.envs.g1_vision.features import (
    BASE_FEATURE_NAMES,
    FeatureHistoryStack,
    compute_u_err_ctrl,
    extract_base_features,
)
from legged_gym.envs.g1_vision.highlevel_policy import (
    HighLevelCommandAdapter,
    load_bc_checkpoint,
    normalize_features,
)
from legged_gym.envs.g1_vision.scenarios import apply_scenario_to_follower, sample_episode_scenario
from legged_gym.utils import get_args, task_registry

from g1_pid_module import (
    DotsSplinePidFollower,
    draw_camera_debug,
    draw_command_arrows,
    draw_path_and_dashes,
    draw_tracking_points,
    get_local_pose_rpy,
    perturb_initial_pose,
    reset_done_envs,
)


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--task", type=str, default="g1_vision")
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=12000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no_viewer", action="store_true", default=False)
    parser.add_argument("--draw_debug", action="store_true", default=False)
    parser.add_argument("--scenario_preset", type=str, default="mixed", choices=["basic", "mixed", "hard", "extreme"])
    parser.add_argument("--history_steps", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--compare_teacher", action="store_true", default=False)
    parser.add_argument(
        "--bc_output_mode",
        type=str,
        default="direct",
        choices=["direct", "adapter"],
        help="Interpret BC output as final command (direct) or as adapter target (adapter).",
    )
    parser.add_argument(
        "--use_recovery_override",
        action="store_true",
        default=False,
        help="Apply teacher-style recovery override to BC output.",
    )

    play_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()

    base_args.task = play_args.task
    base_args.seed = play_args.seed
    base_args.num_envs = play_args.num_envs
    if play_args.load_run is not None:
        base_args.load_run = play_args.load_run
    if play_args.checkpoint is not None:
        base_args.checkpoint = play_args.checkpoint
    if play_args.no_viewer:
        base_args.headless = True

    return play_args, base_args


def configure_env_cfg(env_cfg, num_envs):
    env_cfg.env.num_envs = max(1, int(num_envs))
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 60.0

    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "resampling_time"):
            env_cfg.commands.resampling_time = 999999.0
        if hasattr(env_cfg.commands, "heading_command"):
            env_cfg.commands.heading_command = False


def _sync_teacher_vision_state(teacher, src):
    teacher.vision_u_err[:] = src.vision_u_err
    teacher.vision_u_err_la[:] = src.vision_u_err_la
    teacher.vision_slope[:] = src.vision_slope
    teacher.n_visible[:] = src.n_visible


def _apply_recovery_override_to_target(follower, target_vx_wz: torch.Tensor) -> torch.Tensor:
    target = target_vx_wz.clone()

    # Use vision-derived nominal turn direction for recovery, not BC output sign.
    # This follows teacher logic and avoids opposite-turn lock-in under poor visibility.
    u_err_ctrl = compute_u_err_ctrl(follower)
    w_nom = -follower.k_u * u_err_ctrl - follower.k_slope * follower.vision_slope

    turn_sign = torch.sign(w_nom)
    turn_sign = torch.where(turn_sign == 0.0, torch.sign(follower.w_cmd_prev), turn_sign)
    turn_sign = torch.where(turn_sign == 0.0, torch.ones_like(turn_sign), turn_sign)

    recover_v = torch.full_like(target[:, 0], float(follower.recover_vx))
    recover_w = torch.full_like(target[:, 1], float(follower.recover_wz)) * turn_sign

    target[:, 0] = torch.where(follower.in_recovery, recover_v, target[:, 0])
    target[:, 1] = torch.where(follower.in_recovery, recover_w, target[:, 1])

    no_vis = follower.n_visible < 0.5
    target[:, 0] = torch.where(no_vis, torch.full_like(target[:, 0], 0.10), target[:, 0])
    target[:, 1] = torch.where(no_vis, 0.95 * follower.w_cmd_prev, target[:, 1])
    return target


def _build_direct_commands(follower, target_vx_wz: torch.Tensor) -> torch.Tensor:
    vx = torch.clamp(target_vx_wz[:, 0], float(follower.vx_min), float(follower.vx_max))
    wz = torch.clamp(target_vx_wz[:, 1], float(follower.wz_min), float(follower.wz_max))

    follower.v_cmd_prev[:] = vx
    follower.w_cmd_prev[:] = wz
    follower.v_cmd_hold[:] = vx
    follower.w_cmd_hold[:] = wz
    follower.v_cmd_start[:] = vx
    follower.w_cmd_start[:] = wz
    follower.v_cmd_goal[:] = vx
    follower.w_cmd_goal[:] = wz
    follower.interp_countdown.zero_()

    commands = torch.zeros((follower.num_envs, 4), device=follower.device)
    commands[:, 0] = vx
    commands[:, 1] = 0.0
    commands[:, 2] = wz
    commands[:, 3] = 0.0
    return commands


def _reset_low_level_memory_if_recurrent(ppo_runner, env_ids: torch.Tensor):
    actor_critic = ppo_runner.alg.actor_critic
    if hasattr(actor_critic, "memory_a"):
        memory_a = actor_critic.memory_a
        if getattr(memory_a, "hidden_states", None) is not None:
            memory_a.reset(env_ids)


def run(play_args, base_args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=base_args.task)
    configure_env_cfg(env_cfg=env_cfg, num_envs=base_args.num_envs)
    env, _ = task_registry.make_env(name=base_args.task, args=base_args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=base_args.task,
        args=base_args,
        train_cfg=train_cfg,
    )
    low_level_policy = ppo_runner.get_inference_policy(device=env.device)

    model, feat_mean, feat_std, ckpt_feature_names, _ckpt = load_bc_checkpoint(
        ckpt_path=play_args.bc_checkpoint,
        device=env.device,
    )
    base_dim = len(BASE_FEATURE_NAMES)
    inferred_hist = max(1, int(feat_mean.numel() // base_dim))
    history_steps = inferred_hist if int(play_args.history_steps) <= 0 else int(play_args.history_steps)
    if history_steps * base_dim != int(feat_mean.numel()):
        raise ValueError(
            f"Feature dim mismatch: checkpoint expects {int(feat_mean.numel())}, "
            f"but history_steps={history_steps} -> {history_steps * base_dim}"
        )
    print(
        f"[bc_play] mode={play_args.bc_output_mode} "
        f"recovery_override={int(bool(play_args.use_recovery_override))} "
        f"history_steps={history_steps}"
    )

    follower = DotsSplinePidFollower(
        num_envs=env.num_envs,
        device=env.device,
        env_dt=env.dt,
        seed=play_args.seed,
    )
    follower.hold_steps = 10
    follower.command_interp_steps = 5
    bc_cmd_adapter = HighLevelCommandAdapter(follower)

    teacher_follower = None
    if play_args.compare_teacher:
        teacher_follower = DotsSplinePidFollower(
            num_envs=env.num_envs,
            device=env.device,
            env_dt=env.dt,
            seed=play_args.seed + 999,
        )
        teacher_follower.hold_steps = 10
        teacher_follower.command_interp_steps = 5

    feature_stack = FeatureHistoryStack(
        num_envs=env.num_envs,
        feature_dim=base_dim,
        history_steps=history_steps,
        device=env.device,
    )

    scenario_rng = np.random.default_rng(play_args.seed)
    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    scenario = sample_episode_scenario(
        rng=scenario_rng,
        preset=play_args.scenario_preset,
        episode_idx=0,
    )
    apply_scenario_to_follower(follower, scenario)
    if teacher_follower is not None:
        apply_scenario_to_follower(teacher_follower, scenario)
        teacher_follower.reset_env_state(all_env_ids)

    follower.reset_env_state(all_env_ids)
    feature_stack.reset()
    _reset_low_level_memory_if_recurrent(ppo_runner=ppo_runner, env_ids=all_env_ids)
    perturb_initial_pose(
        env,
        y_range=float(scenario.init_y_range),
        yaw_range=float(scenario.init_yaw_range),
    )
    obs = env.get_observations()

    episode_idx = 0
    step_in_episode = 0
    teacher_mae_sum = 0.0
    teacher_mae_count = 0

    for global_step in range(int(play_args.num_steps)):
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

        base_features = extract_base_features(follower)
        features = feature_stack.update(base_features)
        norm_features = normalize_features(features, mean=feat_mean, std=feat_std)
        with torch.no_grad():
            bc_action = model(norm_features)

        if step_in_episode % max(1, int(follower.hold_steps)) == 0:
            follower._update_recovery_state_vision()

        target_vx_wz = bc_action
        if play_args.use_recovery_override:
            target_vx_wz = _apply_recovery_override_to_target(
                follower=follower, target_vx_wz=target_vx_wz
            )

        if play_args.bc_output_mode == "adapter":
            commands = bc_cmd_adapter.step(target_vx_wz=target_vx_wz, step_idx=step_in_episode)
        else:
            commands = _build_direct_commands(follower=follower, target_vx_wz=target_vx_wz)
        env.commands[:, :] = commands

        teacher_diff_vx = 0.0
        teacher_diff_wz = 0.0
        if teacher_follower is not None:
            _sync_teacher_vision_state(teacher_follower, follower)
            teacher_commands = teacher_follower.compute_upper_command_from_vision(step_idx=step_in_episode)
            teacher_diff_vx = float(torch.mean(torch.abs(commands[:, 0] - teacher_commands[:, 0])).item())
            teacher_diff_wz = float(torch.mean(torch.abs(commands[:, 2] - teacher_commands[:, 2])).item())
            teacher_mae_sum += 0.5 * (teacher_diff_vx + teacher_diff_wz)
            teacher_mae_count += 1

        if play_args.draw_debug:
            draw_path_and_dashes(env, follower)
            draw_tracking_points(env, follower)
            draw_camera_debug(env, follower, z_ground=follower.path_z)
            draw_command_arrows(env, follower, commands)

        obs = env.get_observations()
        with torch.no_grad():
            lower_actions = low_level_policy(obs.detach())
        obs, _, _rews, dones, _infos = env.step(lower_actions.detach())

        if play_args.print_every > 0 and (global_step % play_args.print_every == 0):
            msg = (
                f"[bc_play] step={global_step} ep={episode_idx} ep_step={step_in_episode} "
                f"cmd=({float(commands[0,0].item()):+.3f},{float(commands[0,2].item()):+.3f}) "
                f"n_visible={float(follower.n_visible[0].item()):.1f} "
                f"recovery={int(follower.in_recovery[0].item())}"
            )
            if teacher_follower is not None:
                msg += f" teacher_mae(vx,wz)=({teacher_diff_vx:.4f},{teacher_diff_wz:.4f})"
            print(msg)

        if torch.any(dones):
            done_ids = torch.nonzero(dones).flatten()
            reset_done_envs(env, done_ids, follower)
            feature_stack.reset(done_ids)
            _reset_low_level_memory_if_recurrent(ppo_runner=ppo_runner, env_ids=done_ids)
            if teacher_follower is not None:
                teacher_follower.reset_env_state(done_ids)

            episode_idx += 1
            step_in_episode = 0
            scenario = sample_episode_scenario(
                rng=scenario_rng,
                preset=play_args.scenario_preset,
                episode_idx=episode_idx,
            )
            apply_scenario_to_follower(follower, scenario)
            if teacher_follower is not None:
                apply_scenario_to_follower(teacher_follower, scenario)
                teacher_follower.reset_env_state(all_env_ids)
            perturb_initial_pose(
                env,
                y_range=float(scenario.init_y_range),
                yaw_range=float(scenario.init_yaw_range),
            )
            obs = env.get_observations()
            continue

        step_in_episode += 1

    if teacher_mae_count > 0:
        print(f"[bc_play] mean teacher diff: {teacher_mae_sum / float(teacher_mae_count):.6f}")
    if len(ckpt_feature_names) > 0:
        print(f"[bc_play] checkpoint feature dim={len(ckpt_feature_names)}")


if __name__ == "__main__":
    play_args, base_args = parse_args()
    run(play_args=play_args, base_args=base_args)
