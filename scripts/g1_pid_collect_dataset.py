import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import isaacgym
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.envs.g1_vision.features import (
    BASE_FEATURE_NAMES,
    FeatureHistoryStack,
    build_feature_names,
    extract_base_features,
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


def parse_collect_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--task", type=str, default="g1_vision")
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--dataset_dir", type=str, default="legged_gym/datasets/g1_vision")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--no_viewer", action="store_true", default=False)
    parser.add_argument("--draw_debug", action="store_true", default=False)
    parser.add_argument("--scenario_preset", type=str, default="mixed", choices=["basic", "mixed", "hard"])
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--max_steps_per_episode", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--stop_on_success", action="store_true", default=True)
    parser.add_argument("--no_stop_on_success", action="store_false", dest="stop_on_success")
    parser.add_argument("--success_progress_ratio", type=float, default=0.98)
    parser.add_argument("--success_path_dist", type=float, default=0.60)

    collect_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()

    base_args.task = collect_args.task
    base_args.seed = collect_args.seed
    base_args.num_envs = collect_args.num_envs
    if collect_args.load_run is not None:
        base_args.load_run = collect_args.load_run
    if collect_args.checkpoint is not None:
        base_args.checkpoint = collect_args.checkpoint
    if collect_args.no_viewer:
        base_args.headless = True

    return collect_args, base_args


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


def _compute_progress(local_x, prev_local_x):
    if prev_local_x is None:
        return torch.zeros_like(local_x)
    return local_x - prev_local_x


def _compute_path_progress_and_success(
    follower,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    progress_ratio_thr: float,
    path_dist_thr: float,
):
    device = local_x.device
    num_envs = int(local_x.shape[0])
    progress_ratio = torch.zeros(num_envs, device=device, dtype=torch.float32)
    nearest_dist = torch.full((num_envs,), 1e9, device=device, dtype=torch.float32)
    success_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)

    for i in range(num_envs):
        path_xy = follower.path_points[i]
        path_s = follower.path_s[i]
        if path_xy is None or path_s is None or path_xy.shape[0] == 0:
            continue

        pos = torch.stack([local_x[i], local_y[i]], dim=0)
        delta = path_xy - pos[None, :]
        d2 = torch.sum(delta * delta, dim=1)
        nearest_idx = torch.argmin(d2)
        dmin = torch.sqrt(torch.clamp(d2[nearest_idx], min=0.0))
        s_now = path_s[nearest_idx]
        s_end = torch.clamp(path_s[-1], min=1e-6)

        ratio = torch.clamp(s_now / s_end, 0.0, 1.0)
        progress_ratio[i] = ratio
        nearest_dist[i] = dmin
        success_mask[i] = (ratio >= float(progress_ratio_thr)) & (dmin <= float(path_dist_thr))

    return progress_ratio, nearest_dist, success_mask


def _save_episode_npz(
    out_path: Path,
    feature_names,
    scenario_dict,
    episode_arrays,
    episode_meta,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=episode_arrays["features"].astype(np.float32),
        actions=episode_arrays["actions"].astype(np.float32),
        env_ids=episode_arrays["env_ids"].astype(np.int32),
        episode_ids=episode_arrays["episode_ids"].astype(np.int32),
        step_ids=episode_arrays["step_ids"].astype(np.int32),
        global_step_ids=episode_arrays["global_step_ids"].astype(np.int64),
        progress=episode_arrays["progress"].astype(np.float32),
        path_progress_ratio=episode_arrays["path_progress_ratio"].astype(np.float32),
        path_nearest_dist=episode_arrays["path_nearest_dist"].astype(np.float32),
        cross_track_error=episode_arrays["cross_track_error"].astype(np.float32),
        heading_error=episode_arrays["heading_error"].astype(np.float32),
        u_err_near=episode_arrays["u_err_near"].astype(np.float32),
        u_err_lookahead=episode_arrays["u_err_lookahead"].astype(np.float32),
        u_err_ctrl=episode_arrays["u_err_ctrl"].astype(np.float32),
        slope=episode_arrays["slope"].astype(np.float32),
        n_visible=episode_arrays["n_visible"].astype(np.float32),
        in_recovery=episode_arrays["in_recovery"].astype(np.float32),
        feature_names=np.asarray(feature_names),
        action_names=np.asarray(["vx", "wz"]),
        scenario_id=np.asarray([scenario_dict["scenario_id"]]),
        scenario_level=np.asarray([scenario_dict["level"]]),
        scenario_preset=np.asarray([scenario_dict["preset"]]),
        scenario_json=np.asarray([json.dumps(scenario_dict, sort_keys=True)]),
        episode_meta_json=np.asarray([json.dumps(episode_meta, sort_keys=True)]),
    )


def _concat_episode_lists(episode_lists):
    out = {}
    for k, v in episode_lists.items():
        out[k] = np.concatenate(v, axis=0) if len(v) > 0 else np.zeros((0,), dtype=np.float32)
    return out


def _detect_next_episode_index(rollouts_dir: Path) -> int:
    existing = sorted(rollouts_dir.glob("episode_*.npz"))
    if len(existing) == 0:
        return 0

    max_idx = -1
    for p in existing:
        try:
            ep_idx = int(p.stem.split("_")[-1])
            max_idx = max(max_idx, ep_idx)
        except Exception:
            continue
    return max(0, max_idx + 1)


def _reset_low_level_memory_if_recurrent(ppo_runner, env_ids: torch.Tensor):
    actor_critic = ppo_runner.alg.actor_critic
    if hasattr(actor_critic, "memory_a"):
        memory_a = actor_critic.memory_a
        if getattr(memory_a, "hidden_states", None) is not None:
            memory_a.reset(env_ids)


def collect_dataset(collect_args, base_args):
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

    follower = DotsSplinePidFollower(
        num_envs=env.num_envs,
        device=env.device,
        env_dt=env.dt,
        seed=collect_args.seed,
    )

    # Keep teacher upper command update at 5Hz (0.2s if env.dt=0.02).
    follower.hold_steps = 10
    follower.command_interp_steps = 5

    scenario_rng = np.random.default_rng(collect_args.seed)
    history_steps = max(1, int(collect_args.history_steps))
    feature_stack = FeatureHistoryStack(
        num_envs=env.num_envs,
        feature_dim=len(BASE_FEATURE_NAMES),
        history_steps=history_steps,
        device=env.device,
    )
    feature_names = build_feature_names(history_steps)

    dataset_dir = Path(collect_args.dataset_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    rollouts_dir = dataset_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.jsonl"
    start_episode_idx = _detect_next_episode_index(rollouts_dir)
    if start_episode_idx > 0:
        print(
            f"[collect] existing rollouts detected. "
            f"start episode index = {start_episode_idx}"
        )

    run_meta = {
        "created_at": datetime.now().isoformat(),
        "task": base_args.task,
        "load_run": base_args.load_run,
        "checkpoint": base_args.checkpoint,
        "seed": collect_args.seed,
        "scenario_preset": collect_args.scenario_preset,
        "history_steps": history_steps,
        "feature_names": feature_names,
        "action_names": ["vx", "wz"],
        "num_envs": env.num_envs,
        "dt": float(env.dt),
        "hold_steps": int(follower.hold_steps),
        "command_interp_steps": int(follower.command_interp_steps),
        "stop_on_success": bool(collect_args.stop_on_success),
        "success_progress_ratio": float(collect_args.success_progress_ratio),
        "success_path_dist": float(collect_args.success_path_dist),
    }
    (dataset_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    global_step = 0
    max_steps = (
        int(env.max_episode_length)
        if int(collect_args.max_steps_per_episode) <= 0
        else int(collect_args.max_steps_per_episode)
    )

    for local_episode_idx in range(int(collect_args.num_episodes)):
        episode_idx = start_episode_idx + local_episode_idx
        scenario = sample_episode_scenario(
            rng=scenario_rng,
            preset=collect_args.scenario_preset,
            episode_idx=episode_idx,
        )
        # Important: each dataset episode should start from a fresh env state.
        # Otherwise early-success break can leak episode_length_buf and trigger timeout in next episode.
        env.reset_idx(all_env_ids)
        _reset_low_level_memory_if_recurrent(ppo_runner=ppo_runner, env_ids=all_env_ids)

        apply_scenario_to_follower(follower, scenario)
        follower.reset_env_state(all_env_ids)
        feature_stack.reset()
        perturb_initial_pose(
            env,
            y_range=float(scenario.init_y_range),
            yaw_range=float(scenario.init_yaw_range),
        )
        obs = env.get_observations()

        episode_lists = {
            "features": [],
            "actions": [],
            "env_ids": [],
            "episode_ids": [],
            "step_ids": [],
            "global_step_ids": [],
            "progress": [],
            "path_progress_ratio": [],
            "path_nearest_dist": [],
            "cross_track_error": [],
            "heading_error": [],
            "u_err_near": [],
            "u_err_lookahead": [],
            "u_err_ctrl": [],
            "slope": [],
            "n_visible": [],
            "in_recovery": [],
        }
        prev_local_x = None
        terminated_by_done = False
        terminated_by_success = False

        for step_idx in range(max_steps):
            local_x, local_y, _roll, _pitch, yaw = get_local_pose_rpy(env)
            base_z = env.root_states[:, 2]

            follower.update_perception(
                local_x=local_x,
                local_y=local_y,
                base_z=base_z,
                roll=_roll,
                pitch=_pitch,
                yaw=yaw,
                env_origins=env.env_origins,
            )

            base_features = extract_base_features(follower)
            features = feature_stack.update(base_features)
            commands = follower.compute_upper_command_from_vision(step_idx=step_idx)
            teacher_actions = torch.stack([commands[:, 0], commands[:, 2]], dim=1)

            progress = _compute_progress(local_x=local_x, prev_local_x=prev_local_x)
            prev_local_x = local_x.clone()
            path_prog_ratio, path_nearest_dist, success_mask = _compute_path_progress_and_success(
                follower=follower,
                local_x=local_x,
                local_y=local_y,
                progress_ratio_thr=float(collect_args.success_progress_ratio),
                path_dist_thr=float(collect_args.success_path_dist),
            )

            env.commands[:, :] = commands

            episode_lists["features"].append(features.detach().cpu().numpy())
            episode_lists["actions"].append(teacher_actions.detach().cpu().numpy())
            episode_lists["env_ids"].append(np.arange(env.num_envs, dtype=np.int32))
            episode_lists["episode_ids"].append(np.full((env.num_envs,), episode_idx, dtype=np.int32))
            episode_lists["step_ids"].append(np.full((env.num_envs,), step_idx, dtype=np.int32))
            episode_lists["global_step_ids"].append(
                np.full((env.num_envs,), global_step, dtype=np.int64)
            )
            episode_lists["progress"].append(progress.detach().cpu().numpy())
            episode_lists["path_progress_ratio"].append(path_prog_ratio.detach().cpu().numpy())
            episode_lists["path_nearest_dist"].append(path_nearest_dist.detach().cpu().numpy())
            episode_lists["cross_track_error"].append(local_y.detach().cpu().numpy())
            episode_lists["heading_error"].append(yaw.detach().cpu().numpy())
            episode_lists["u_err_near"].append(follower.vision_u_err.detach().cpu().numpy())
            episode_lists["u_err_lookahead"].append(follower.vision_u_err_la.detach().cpu().numpy())
            episode_lists["u_err_ctrl"].append(base_features[:, 2].detach().cpu().numpy())
            episode_lists["slope"].append(follower.vision_slope.detach().cpu().numpy())
            episode_lists["n_visible"].append(follower.n_visible.detach().cpu().numpy())
            episode_lists["in_recovery"].append(
                follower.in_recovery.to(torch.float32).detach().cpu().numpy()
            )

            if collect_args.draw_debug:
                draw_path_and_dashes(env, follower)
                draw_tracking_points(env, follower)
                draw_camera_debug(env, follower, z_ground=follower.path_z)
                draw_command_arrows(env, follower, commands)

            obs = env.get_observations()
            with torch.no_grad():
                lower_actions = low_level_policy(obs.detach())
            obs, _, _rews, dones, _infos = env.step(lower_actions.detach())
            global_step += 1

            if torch.any(dones):
                done_ids = torch.nonzero(dones).flatten()
                timeout_mask = torch.zeros_like(dones, dtype=torch.bool)
                if isinstance(_infos, dict) and ("time_outs" in _infos):
                    timeout_mask = _infos["time_outs"].to(env.device).bool()
                done_timeout = int(torch.sum(timeout_mask[done_ids]).item())
                done_fail = int(done_ids.numel()) - done_timeout

                reset_done_envs(env, done_ids, follower)
                feature_stack.reset(done_ids)
                terminated_by_done = True
                print(
                    f"[collect] episode={episode_idx} step={step_idx} "
                    f"done_envs={int(done_ids.numel())} fail={done_fail} timeout={done_timeout}"
                )
                break

            if bool(collect_args.stop_on_success) and torch.any(success_mask):
                terminated_by_success = True
                break

            if collect_args.print_every > 0 and (step_idx % collect_args.print_every == 0):
                print(
                    f"[collect] episode={episode_idx} step={step_idx} "
                    f"n_visible={float(follower.n_visible[0].item()):.1f} "
                    f"cmd=({float(commands[0, 0].item()):+.3f},{float(commands[0, 2].item()):+.3f}) "
                    f"path_ratio={float(path_prog_ratio[0].item()):.3f} "
                    f"path_dist={float(path_nearest_dist[0].item()):.3f}"
                )

        episode_arrays = _concat_episode_lists(episode_lists)
        num_samples = int(episode_arrays["features"].shape[0])
        episode_file = rollouts_dir / f"episode_{episode_idx:06d}.npz"
        scenario_dict = scenario.to_dict()
        episode_meta = {
            "episode_idx": episode_idx,
            "num_samples": num_samples,
            "num_steps": int(num_samples // env.num_envs if env.num_envs > 0 else 0),
            "terminated_by_done": bool(terminated_by_done),
            "terminated_by_success": bool(terminated_by_success),
            "terminated_by_max_steps": bool((not terminated_by_done) and (not terminated_by_success)),
            "max_steps_per_episode": int(max_steps),
        }
        _save_episode_npz(
            out_path=episode_file,
            feature_names=feature_names,
            scenario_dict=scenario_dict,
            episode_arrays=episode_arrays,
            episode_meta=episode_meta,
        )

        with manifest_path.open("a") as f:
            manifest_row = {
                "episode_idx": episode_idx,
                "file": str(episode_file),
                "num_samples": num_samples,
                "num_steps": episode_meta["num_steps"],
                "terminated_by_done": episode_meta["terminated_by_done"],
                "terminated_by_success": episode_meta["terminated_by_success"],
                "terminated_by_max_steps": episode_meta["terminated_by_max_steps"],
                "scenario_id": scenario.scenario_id,
                "scenario_level": scenario.level,
                "scenario_preset": scenario.preset,
            }
            f.write(json.dumps(manifest_row, sort_keys=True) + "\n")

        print(
            f"[collect] saved episode={episode_idx} "
            f"samples={num_samples} done={terminated_by_done} "
            f"success={terminated_by_success} "
            f"max_steps_end={episode_meta['terminated_by_max_steps']} file={episode_file.name}"
        )

    print(f"[collect] dataset saved to: {dataset_dir}")


if __name__ == "__main__":
    collect_args, base_args = parse_collect_args()
    collect_dataset(collect_args=collect_args, base_args=base_args)
