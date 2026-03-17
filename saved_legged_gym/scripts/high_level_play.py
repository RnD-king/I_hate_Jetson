import argparse
import os
import sys

import isaacgym
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
    parser.add_argument("--disable_high_fail", action="store_true", default=False)

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
    if hl_args.disable_high_fail:
        env_cfg.high_level.fail_ey = 1e9
        env_cfg.high_level.fail_epsi = 1e9
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


def run_rollout(env, obs, actions_fn, max_steps, print_every):
    sum_done_rate = 0.0
    sum_total_done_rate = 0.0
    sum_base_done_rate = 0.0
    sum_high_fail_rate = 0.0
    sum_vx = 0.0
    sum_wz = 0.0

    for i in range(max_steps):
        actions = actions_fn(obs)
        obs, _, rews, dones, infos = env.step(actions.detach())
        done_rate = dones.float().mean().item()
        episode = infos.get("episode", {})

        base_done_rate = infos.get("base_done_rate", episode.get("metric_base_done_rate", 0.0))
        high_fail_rate = infos.get("high_fail_rate", episode.get("metric_high_fail_rate", 0.0))
        total_done_rate = infos.get("total_done_rate", episode.get("metric_total_done_rate", done_rate))
        vx_mean = infos.get("high_cmd_vx_mean", 0.0)
        wz_mean = infos.get("high_cmd_wz_mean", 0.0)

        sum_done_rate += done_rate
        sum_total_done_rate += total_done_rate
        sum_base_done_rate += base_done_rate
        sum_high_fail_rate += high_fail_rate
        sum_vx += vx_mean
        sum_wz += wz_mean

        if i % max(1, print_every) == 0:
            print(
                f"step={i:05d} "
                f"rew_mean={rews.mean().item():+.4f} "
                f"cmd(vx,wz)=({vx_mean:+.3f}, {wz_mean:+.3f}) "
                f"ey_mean={episode.get('metric_ey_abs_mean', 0.0):.4f} "
                f"done={done_rate:.3f} "
                f"base_done={base_done_rate:.3f} "
                f"high_fail={high_fail_rate:.3f} "
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
    if base_args.task != "g1_highlevel":
        raise ValueError(
            f"high_level_play.py is intended for --task g1_highlevel, got: {base_args.task}"
        )

    env_cfg, train_cfg = task_registry.get_cfgs(name=base_args.task)
    apply_eval_overrides(env_cfg=env_cfg, num_envs_cap=hl_args.num_envs_cap)
    apply_wrapper_overrides(env_cfg=env_cfg, hl_args=hl_args)

    env, _ = task_registry.make_env(name=base_args.task, args=base_args, env_cfg=env_cfg)
    obs = env.get_observations()

    max_steps = hl_args.steps if hl_args.steps > 0 else 10 * int(env.max_episode_length)

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
    )


if __name__ == "__main__":
    hl_args, base_args = parse_high_level_args()
    play_high_level(hl_args, base_args)
