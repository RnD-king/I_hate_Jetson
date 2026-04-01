import argparse
import os
import sys

import isaacgym
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import export_policy_as_jit, get_args, task_registry


def parse_high_level_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", type=str, default="policy", choices=["policy", "fixed"])
    parser.add_argument("--fixed_vx", type=float, default=0.4)
    parser.add_argument("--fixed_wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--num_envs_cap", type=int, default=100)
    parser.add_argument("--export_policy", action="store_true", default=False)
    parser.add_argument("--no_resume", action="store_true", default=False)
    parser.add_argument("--override_hold_steps", type=int, default=None)
    parser.add_argument("--override_dv_max", type=float, default=None)
    parser.add_argument("--override_dw_max", type=float, default=None)
    parser.add_argument("--override_low_level_decimation", type=int, default=None)
    parser.add_argument("--override_vx_min", type=float, default=None)
    parser.add_argument("--override_vx_max", type=float, default=None)
    parser.add_argument("--override_wz_min", type=float, default=None)
    parser.add_argument("--override_wz_max", type=float, default=None)
    parser.add_argument("--override_low_level_checkpoint_path", type=str, default=None)
    parser.add_argument("--disable_high_fail", action="store_true", default=False)
    parser.add_argument("--focus_env_id", type=int, default=0)
    parser.add_argument("--focus_cmd_print_every", type=int, default=0)
    parser.add_argument(
        "--print_all_fail_reasons_every",
        type=int,
        default=1,
        help="Print all-env fail reasons every N steps (<=0 disables).",
    )
    parser.add_argument("--disable_focus_marker", action="store_true", default=False)
    parser.add_argument("--focus_marker_z_offset", type=float, default=0.62)

    hl_args, remaining = parser.parse_known_args()

    # Delegate the rest of CLI parsing to legged_gym's IsaacGym parser.
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    return hl_args, base_args


def apply_eval_overrides(env_cfg, num_envs_cap):
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max(1, num_envs_cap))
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True


def apply_wrapper_overrides(env_cfg, hl_args):
    if hl_args.override_hold_steps is not None:
        env_cfg.high_level.hold_steps = max(1, int(hl_args.override_hold_steps))
    if hl_args.override_dv_max is not None:
        env_cfg.high_level.dv_max = float(hl_args.override_dv_max)
    if hl_args.override_dw_max is not None:
        env_cfg.high_level.dw_max = float(hl_args.override_dw_max)
    if hl_args.override_low_level_decimation is not None:
        env_cfg.high_level.low_level_decimation = max(1, int(hl_args.override_low_level_decimation))
    if hl_args.override_vx_min is not None:
        env_cfg.high_level.vx_min = float(hl_args.override_vx_min)
    if hl_args.override_vx_max is not None:
        env_cfg.high_level.vx_max = float(hl_args.override_vx_max)
    if hl_args.override_wz_min is not None:
        env_cfg.high_level.wz_min = float(hl_args.override_wz_min)
    if hl_args.override_wz_max is not None:
        env_cfg.high_level.wz_max = float(hl_args.override_wz_max)
    if hl_args.override_low_level_checkpoint_path is not None:
        env_cfg.high_level.low_level_checkpoint_path = str(hl_args.override_low_level_checkpoint_path)

    if env_cfg.high_level.vx_min >= env_cfg.high_level.vx_max:
        raise ValueError(
            f"Invalid vx range: [{env_cfg.high_level.vx_min}, {env_cfg.high_level.vx_max}]"
        )
    if env_cfg.high_level.wz_min >= env_cfg.high_level.wz_max:
        raise ValueError(
            f"Invalid wz range: [{env_cfg.high_level.wz_min}, {env_cfg.high_level.wz_max}]"
        )
    if hl_args.disable_high_fail:
        env_cfg.high_level.fail_ey = 1e9
        env_cfg.high_level.fail_epsi = 1e9
        if hasattr(env_cfg.high_level, "fail_roll"):
            env_cfg.high_level.fail_roll = 1e9
        if hasattr(env_cfg.high_level, "fail_pitch"):
            env_cfg.high_level.fail_pitch = 1e9
        env_cfg.high_level.fail_penalty = 0.0


def fixed_cmd_to_high_action(env, fixed_vx, fixed_wz):
    vx = max(env.high_cfg.vx_min, min(env.high_cfg.vx_max, fixed_vx))
    wz = max(env.high_cfg.wz_min, min(env.high_cfg.wz_max, fixed_wz))

    vx_den = max(env.high_cfg.vx_max - env.high_cfg.vx_min, 1e-6)
    wz_den = max(env.high_cfg.wz_max - env.high_cfg.wz_min, 1e-6)

    a_vx = 2.0 * (vx - env.high_cfg.vx_min) / vx_den - 1.0
    a_wz = 2.0 * (wz - env.high_cfg.wz_min) / wz_den - 1.0

    action = torch.tensor([a_vx, a_wz], device=env.device, dtype=torch.float)
    return action.unsqueeze(0).repeat(env.num_envs, 1), vx, wz


def maybe_load_policy(env, task_name, base_args, train_cfg, export_policy):
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=task_name,
        args=base_args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)

    if export_policy:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    return policy


def _draw_focus_env_marker(env, focus_env_id, z_offset):
    if env.viewer is None:
        return
    if focus_env_id < 0 or focus_env_id >= env.num_envs:
        return

    env.gym.clear_lines(env.viewer)

    base_pos = env.root_states[focus_env_id, 0:3].detach().cpu().numpy()
    x = float(base_pos[0])
    y = float(base_pos[1])
    z = float(base_pos[2] + z_offset)

    half_len = 0.06
    color = np.array([1.0, 0.9, 0.1], dtype=np.float32)

    px1 = np.array([x - half_len, y, z], dtype=np.float32)
    px2 = np.array([x + half_len, y, z], dtype=np.float32)
    py1 = np.array([x, y - half_len, z], dtype=np.float32)
    py2 = np.array([x, y + half_len, z], dtype=np.float32)
    pz1 = np.array([x, y, z - half_len], dtype=np.float32)
    pz2 = np.array([x, y, z + half_len], dtype=np.float32)

    env_handle = env.envs[focus_env_id]
    env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([px1, px2]).astype(np.float32), color)
    env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([py1, py2]).astype(np.float32), color)
    env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([pz1, pz2]).astype(np.float32), color)


def run_rollout(
    env,
    obs,
    actions_fn,
    max_steps,
    print_every,
    focus_env_id,
    focus_cmd_print_every,
    print_all_fail_reasons_every,
    disable_focus_marker,
    focus_marker_z_offset,
):
    sum_done_rate = 0.0
    sum_total_done_rate = 0.0
    sum_base_done_rate = 0.0
    sum_high_fail_rate = 0.0
    sum_vx = 0.0
    sum_wz = 0.0
    hold_steps = max(1, int(env.high_cfg.hold_steps))

    for i in range(max_steps):
        actions = actions_fn(obs)
        obs, _, rews, dones, infos = env.step(actions.detach())
        done_rate = dones.float().mean().item()
        episode = infos.get("episode", {})

        base_done_rate = infos.get("base_done_rate", episode.get("metric_base_done_rate", 0.0))
        high_fail_rate = infos.get("high_fail_rate", episode.get("metric_high_fail_rate", 0.0))
        fail_ey_rate = infos.get("fail_ey_rate", episode.get("metric_fail_ey_rate", 0.0))
        fail_epsi_rate = infos.get("fail_epsi_rate", episode.get("metric_fail_epsi_rate", 0.0))
        fail_roll_rate = infos.get("fail_roll_rate", episode.get("metric_fail_roll_rate", 0.0))
        fail_pitch_rate = infos.get("fail_pitch_rate", episode.get("metric_fail_pitch_rate", 0.0))
        total_done_rate = infos.get("total_done_rate", episode.get("metric_total_done_rate", done_rate))
        vx_mean = infos.get("high_cmd_vx_mean", 0.0)
        wz_mean = infos.get("high_cmd_wz_mean", 0.0)

        sum_done_rate += done_rate
        sum_total_done_rate += total_done_rate
        sum_base_done_rate += base_done_rate
        sum_high_fail_rate += high_fail_rate
        sum_vx += vx_mean
        sum_wz += wz_mean

        base_done_mask_dbg = getattr(
            env, "debug_last_base_done_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        base_contact_mask_dbg = getattr(
            env, "debug_last_base_contact_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        base_roll_mask_dbg = getattr(
            env, "debug_last_base_roll_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        base_pitch_mask_dbg = getattr(
            env, "debug_last_base_pitch_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        base_timeout_mask_dbg = getattr(
            env, "debug_last_base_timeout_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        high_fail_mask_dbg = getattr(
            env, "debug_last_high_fail_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        fail_ey_mask_dbg = getattr(
            env, "debug_last_fail_ey_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        fail_epsi_mask_dbg = getattr(
            env, "debug_last_fail_epsi_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        fail_roll_mask_dbg = getattr(
            env, "debug_last_fail_roll_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        fail_pitch_mask_dbg = getattr(
            env, "debug_last_fail_pitch_mask", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )

        if print_all_fail_reasons_every > 0:
            done_mask_dbg = dones.bool()
            base_only_mask = base_done_mask_dbg & (~high_fail_mask_dbg)
            high_only_mask = high_fail_mask_dbg & (~base_done_mask_dbg)
            both_mask = base_done_mask_dbg & high_fail_mask_dbg

            num_done = int(done_mask_dbg.sum().item())
            num_base_only = int(base_only_mask.sum().item())
            num_high_only = int(high_only_mask.sum().item())
            num_both = int(both_mask.sum().item())
            num_base_contact = int(base_contact_mask_dbg.sum().item())
            num_base_roll = int(base_roll_mask_dbg.sum().item())
            num_base_pitch = int(base_pitch_mask_dbg.sum().item())
            num_base_timeout = int(base_timeout_mask_dbg.sum().item())
            num_fail_ey = int(fail_ey_mask_dbg.sum().item())
            num_fail_epsi = int(fail_epsi_mask_dbg.sum().item())
            num_fail_roll = int(fail_roll_mask_dbg.sum().item())
            num_fail_pitch = int(fail_pitch_mask_dbg.sum().item())
            should_print_summary = (i % max(1, print_all_fail_reasons_every) == 0) or (num_done > 0)

            if should_print_summary:
                print(
                    f"[all_env_fail] step={i:05d} "
                    f"done={num_done}/{env.num_envs} "
                    f"base_only={num_base_only} high_only={num_high_only} both={num_both} "
                    f"base_split(contact,roll,pitch,timeout)=({num_base_contact},{num_base_roll},{num_base_pitch},{num_base_timeout}) "
                    f"high_split(ey,epsi,roll,pitch)=({num_fail_ey},{num_fail_epsi},{num_fail_roll},{num_fail_pitch})"
                )

            if num_done > 0:
                done_ids = torch.nonzero(done_mask_dbg, as_tuple=False).flatten().tolist()
                reason_tokens = []
                for env_id in done_ids:
                    parts = []
                    if bool(base_done_mask_dbg[env_id].item()):
                        base_parts = []
                        if bool(base_contact_mask_dbg[env_id].item()):
                            base_parts.append("contact")
                        if bool(base_roll_mask_dbg[env_id].item()):
                            base_parts.append("roll")
                        if bool(base_pitch_mask_dbg[env_id].item()):
                            base_parts.append("pitch")
                        if bool(base_timeout_mask_dbg[env_id].item()):
                            base_parts.append("timeout")
                        if not base_parts:
                            base_parts.append("unknown")
                        parts.append("base:" + "+".join(base_parts))
                    if bool(high_fail_mask_dbg[env_id].item()):
                        high_parts = []
                        if bool(fail_ey_mask_dbg[env_id].item()):
                            high_parts.append("ey")
                        if bool(fail_epsi_mask_dbg[env_id].item()):
                            high_parts.append("epsi")
                        if bool(fail_roll_mask_dbg[env_id].item()):
                            high_parts.append("roll")
                        if bool(fail_pitch_mask_dbg[env_id].item()):
                            high_parts.append("pitch")
                        if not high_parts:
                            high_parts.append("unknown")
                        parts.append("high:" + "+".join(high_parts))
                    if not parts:
                        parts.append("unknown")
                    reason_tokens.append(f"{env_id}:{'|'.join(parts)}")
                print("[all_env_fail_detail] " + " ".join(reason_tokens))

        if focus_cmd_print_every > 0 and i % max(1, focus_cmd_print_every) == 0:
            exec_cmd = getattr(
                env, "debug_last_exec_cmd", torch.zeros(env.num_envs, 2, device=env.device)
            )
            exec_vx = float(exec_cmd[focus_env_id, 0].item())
            exec_wz = float(exec_cmd[focus_env_id, 1].item())
            cmd_vx = float(env.cmd_hold[focus_env_id, 0].item())
            cmd_wz = float(env.cmd_hold[focus_env_id, 1].item())
            base_vx = float(env.base_lin_vel[focus_env_id, 0].item())
            base_wz = float(env.base_ang_vel[focus_env_id, 2].item())
            ey_focus = float(getattr(env, "gt_ey", torch.zeros(1, device=env.device))[focus_env_id].item())
            epsi_focus = float(getattr(env, "gt_epsi", torch.zeros(1, device=env.device))[focus_env_id].item())
            roll_focus = float(getattr(env, "rpy", torch.zeros(env.num_envs, 3, device=env.device))[focus_env_id, 0].item())
            pitch_focus = float(getattr(env, "rpy", torch.zeros(env.num_envs, 3, device=env.device))[focus_env_id, 1].item())
            fail_ey = int(abs(ey_focus) > float(getattr(env.high_cfg, "fail_ey", 1e9)))
            fail_epsi = int(abs(epsi_focus) > float(getattr(env.high_cfg, "fail_epsi", 1e9)))
            fail_roll = int(abs(roll_focus) > float(getattr(env.high_cfg, "fail_roll", 1e9)))
            fail_pitch = int(abs(pitch_focus) > float(getattr(env.high_cfg, "fail_pitch", 1e9)))
            high_ctr = int(env.high_step_counter)
            cmd_update = int(((high_ctr - 1) % hold_steps) == 0)
            block_idx = (high_ctr - 1) // hold_steps
            done_focus = int(bool(dones[focus_env_id].item()))
            base_done_focus = int(bool(base_done_mask_dbg[focus_env_id].item()))
            high_fail_focus = int(bool(high_fail_mask_dbg[focus_env_id].item()))
            print(
                f"[focus_env={focus_env_id:02d}] "
                f"step={i:05d} high_ctr={high_ctr:05d} block={block_idx:05d} "
                f"update={cmd_update} done={done_focus} cause(base,high)=({base_done_focus},{high_fail_focus}) "
                f"cmd_exec(vx,wz)=({exec_vx:+.3f}, {exec_wz:+.3f}) "
                f"cmd_hold(vx,wz)=({cmd_vx:+.3f}, {cmd_wz:+.3f}) "
                f"base(vx,wz)=({base_vx:+.3f}, {base_wz:+.3f}) "
                f"ey={ey_focus:+.3f} epsi={epsi_focus:+.3f} "
                f"roll/pitch=({roll_focus:+.3f},{pitch_focus:+.3f}) "
                f"fail(ey,epsi,roll,pitch)=({fail_ey},{fail_epsi},{fail_roll},{fail_pitch})"
            )

        if not disable_focus_marker:
            _draw_focus_env_marker(
                env=env,
                focus_env_id=focus_env_id,
                z_offset=focus_marker_z_offset,
            )

        if i % max(1, print_every) == 0:
            print(
                f"step={i:05d} "
                f"rew_mean={rews.mean().item():+.4f} "
                f"cmd(vx,wz)=({vx_mean:+.3f}, {wz_mean:+.3f}) "
                f"ey_mean={episode.get('metric_ey_abs_mean', 0.0):.4f} "
                f"done={done_rate:.3f} "
                f"base_done={base_done_rate:.3f} "
                f"high_fail={high_fail_rate:.3f} "
                f"fail_split(ey,epsi,roll,pitch)=({fail_ey_rate:.3f},{fail_epsi_rate:.3f},{fail_roll_rate:.3f},{fail_pitch_rate:.3f}) "
                f"total_done={total_done_rate:.3f} "
                f"success={episode.get('metric_success_rate', 0.0):.3f}"
            )

    n = max(1, max_steps)
    print("=== rollout summary ===")
    print(f"avg_done_rate={sum_done_rate / n:.4f}")
    print(f"avg_base_done_rate={sum_base_done_rate / n:.4f}")
    print(f"avg_high_fail_rate={sum_high_fail_rate / n:.4f}")
    print(f"avg_total_done_rate={sum_total_done_rate / n:.4f}")
    print(f"avg_cmd_vx={sum_vx / n:.4f}")
    print(f"avg_cmd_wz={sum_wz / n:.4f}")
    return obs


def play_high_level(hl_args, base_args):
    if not base_args.task.startswith("g1_highlevel"):
        raise ValueError(
            f"high_level_play.py is intended for --task g1_highlevel*, got: {base_args.task}"
        )

    env_cfg, train_cfg = task_registry.get_cfgs(name=base_args.task)
    apply_eval_overrides(env_cfg=env_cfg, num_envs_cap=hl_args.num_envs_cap)
    apply_wrapper_overrides(env_cfg=env_cfg, hl_args=hl_args)

    env, _ = task_registry.make_env(name=base_args.task, args=base_args, env_cfg=env_cfg)
    obs = env.get_observations()

    max_steps = hl_args.steps if hl_args.steps > 0 else 10 * int(env.max_episode_length)
    focus_env_id = min(max(0, int(hl_args.focus_env_id)), env.num_envs - 1)
    high_step_dt = env.dt * env.high_cfg.low_level_decimation
    cmd_update_period = high_step_dt * env.high_cfg.hold_steps

    print(
        "cfg "
        f"low_ckpt={env.high_cfg.low_level_checkpoint_path} "
        f"hold_steps={env.high_cfg.hold_steps} "
        f"dv_max={env.high_cfg.dv_max:.3f} "
        f"dw_max={env.high_cfg.dw_max:.3f} "
        f"low_decim={env.high_cfg.low_level_decimation} "
        f"fail_ey={env.high_cfg.fail_ey:.3f} "
        f"fail_epsi={env.high_cfg.fail_epsi:.3f} "
        f"sim_device={base_args.sim_device} "
        f"rl_device={base_args.rl_device}"
    )
    print(
        "timing "
        f"sim_dt={env.sim_params.dt:.4f}s "
        f"low_policy_dt={env.dt:.4f}s "
        f"high_step_dt={high_step_dt:.4f}s "
        f"cmd_update_period={cmd_update_period:.4f}s "
        f"(every {env.high_cfg.hold_steps} high-steps)"
    )
    print(
        "focus "
        f"env_id={focus_env_id} env_origin=({env.env_origins[focus_env_id, 0].item():.2f}, "
        f"{env.env_origins[focus_env_id, 1].item():.2f}) "
        f"marker={'off' if hl_args.disable_focus_marker else 'on'} "
        f"cmd_print_every={hl_args.focus_cmd_print_every} "
        f"all_fail_reasons_every={hl_args.print_all_fail_reasons_every}"
    )

    if hl_args.mode == "fixed":
        fixed_actions, vx_clamped, wz_clamped = fixed_cmd_to_high_action(
            env=env,
            fixed_vx=hl_args.fixed_vx,
            fixed_wz=hl_args.fixed_wz,
        )
        print(
            "mode=fixed "
            f"target(vx,wz)=({hl_args.fixed_vx:+.3f}, {hl_args.fixed_wz:+.3f}) "
            f"clamped_cmd=({vx_clamped:+.3f}, {wz_clamped:+.3f}) "
            f"action=({fixed_actions[0, 0].item():+.3f}, {fixed_actions[0, 1].item():+.3f})"
        )
        run_rollout(
            env=env,
            obs=obs,
            actions_fn=lambda _obs: fixed_actions,
            max_steps=max_steps,
            print_every=hl_args.print_every,
            focus_env_id=focus_env_id,
            focus_cmd_print_every=hl_args.focus_cmd_print_every,
            print_all_fail_reasons_every=hl_args.print_all_fail_reasons_every,
            disable_focus_marker=hl_args.disable_focus_marker,
            focus_marker_z_offset=hl_args.focus_marker_z_offset,
        )
        return

    if hl_args.no_resume:
        raise ValueError("--mode policy requires loading a trained checkpoint. Remove --no_resume.")

    policy = maybe_load_policy(
        env=env,
        task_name=base_args.task,
        base_args=base_args,
        train_cfg=train_cfg,
        export_policy=hl_args.export_policy,
    )
    print(
        "mode=policy "
        f"load_run={base_args.load_run} checkpoint={base_args.checkpoint} "
        f"rl_device={base_args.rl_device}"
    )
    run_rollout(
        env=env,
        obs=obs,
        actions_fn=lambda _obs: policy(_obs.detach()),
        max_steps=max_steps,
        print_every=hl_args.print_every,
        focus_env_id=focus_env_id,
        focus_cmd_print_every=hl_args.focus_cmd_print_every,
        print_all_fail_reasons_every=hl_args.print_all_fail_reasons_every,
        disable_focus_marker=hl_args.disable_focus_marker,
        focus_marker_z_offset=hl_args.focus_marker_z_offset,
    )


if __name__ == "__main__":
    hl_args, base_args = parse_high_level_args()
    play_high_level(hl_args, base_args)
