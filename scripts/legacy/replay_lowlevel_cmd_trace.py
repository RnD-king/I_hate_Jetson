import argparse
import re
import sys

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


CMD_PATTERN = re.compile(
    r"cmd\(vx,wz\)=\(\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*,\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*\)"
)
DONE_PATTERN = re.compile(r"done=(\d+)")


def parse_replay_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--preset",
        type=str,
        default="focus_08008_08098",
        choices=["focus_08008_08098"],
        help="Built-in pasted log preset (no file required). Default uses pasted focus log.",
    )
    parser.add_argument("--num_envs_cap", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1, help="If >0, stop after this many total sim steps.")
    parser.add_argument("--focus_env_id", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=1)

    replay_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    return replay_args, base_args


def load_cmd_trace(log_path):
    trace = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = CMD_PATTERN.search(line)
            if m is None:
                continue
            vx = float(m.group(1))
            wz = float(m.group(2))
            trace.append((vx, wz))
    if not trace:
        raise ValueError(f"No cmd(vx,wz)=... entries found in: {log_path}")
    return trace


def load_cmd_trace_from_text(text):
    trace = []
    for line in text.splitlines():
        done_match = DONE_PATTERN.search(line)
        if done_match is not None and int(done_match.group(1)) == 1:
            continue
        m = CMD_PATTERN.search(line)
        if m is None:
            continue
        vx = float(m.group(1))
        wz = float(m.group(2))
        trace.append((vx, wz))
    if not trace:
        raise ValueError("No cmd(vx,wz)=... entries found in preset text")
    return trace


def load_cmd_trace_preset(name):
    if name == "focus_08008_08098":
        preset_text = """
[focus_env=00] step=08008 high_ctr=08010 block=08009 update=1 done=0 cmd(vx,wz)=(+0.000, -0.071) base(vx,wz)=(-0.053, -0.109)
[focus_env=00] step=08009 high_ctr=08011 block=08010 update=1 done=0 cmd(vx,wz)=(+0.000, -0.221) base(vx,wz)=(-0.026, -0.602)
[focus_env=00] step=08010 high_ctr=08012 block=08011 update=1 done=0 cmd(vx,wz)=(+0.000, -0.371) base(vx,wz)=(-0.081, -0.959)
[focus_env=00] step=08011 high_ctr=08013 block=08012 update=1 done=0 cmd(vx,wz)=(+0.000, -0.221) base(vx,wz)=(-0.074, -1.913)
[focus_env=00] step=08012 high_ctr=08014 block=08013 update=1 done=0 cmd(vx,wz)=(+0.000, -0.371) base(vx,wz)=(-0.067, -1.840)
[focus_env=00] step=08013 high_ctr=08015 block=08014 update=1 done=0 cmd(vx,wz)=(+0.000, -0.221) base(vx,wz)=(-0.020, -1.184)
[focus_env=00] step=08014 high_ctr=08016 block=08015 update=1 done=0 cmd(vx,wz)=(+0.000, -0.371) base(vx,wz)=(+0.014, -0.425)
[focus_env=00] step=08015 high_ctr=08017 block=08016 update=1 done=0 cmd(vx,wz)=(+0.000, -0.221) base(vx,wz)=(+0.035, -0.153)
[focus_env=00] step=08016 high_ctr=08018 block=08017 update=1 done=0 cmd(vx,wz)=(+0.000, -0.371) base(vx,wz)=(+0.027, -0.177)
[focus_env=00] step=08017 high_ctr=08019 block=08018 update=1 done=0 cmd(vx,wz)=(+0.000, -0.239) base(vx,wz)=(+0.022, -0.358)
[focus_env=00] step=08018 high_ctr=08020 block=08019 update=1 done=0 cmd(vx,wz)=(+0.000, -0.389) base(vx,wz)=(+0.052, -0.403)
[focus_env=00] step=08019 high_ctr=08021 block=08020 update=1 done=0 cmd(vx,wz)=(+0.000, -0.239) base(vx,wz)=(+0.073, -0.353)
[focus_env=00] step=08020 high_ctr=08022 block=08021 update=1 done=0 cmd(vx,wz)=(+0.000, -0.389) base(vx,wz)=(+0.086, -0.253)
[focus_env=00] step=08021 high_ctr=08023 block=08022 update=1 done=0 cmd(vx,wz)=(+0.000, -0.239) base(vx,wz)=(+0.096, -0.113)
[focus_env=00] step=08022 high_ctr=08024 block=08023 update=1 done=0 cmd(vx,wz)=(+0.000, -0.354) base(vx,wz)=(+0.102, -0.259)
[focus_env=00] step=08023 high_ctr=08025 block=08024 update=1 done=0 cmd(vx,wz)=(+0.000, -0.224) base(vx,wz)=(+0.108, -0.274)
[focus_env=00] step=08024 high_ctr=08026 block=08025 update=1 done=0 cmd(vx,wz)=(+0.000, -0.374) base(vx,wz)=(+0.112, -0.351)
[focus_env=00] step=08025 high_ctr=08027 block=08026 update=1 done=0 cmd(vx,wz)=(+0.000, -0.224) base(vx,wz)=(+0.112, -0.279)
[focus_env=00] step=08026 high_ctr=08028 block=08027 update=1 done=0 cmd(vx,wz)=(+0.000, -0.374) base(vx,wz)=(+0.107, -0.397)
[focus_env=00] step=08027 high_ctr=08029 block=08028 update=1 done=0 cmd(vx,wz)=(+0.000, -0.224) base(vx,wz)=(+0.103, -0.404)
[focus_env=00] step=08028 high_ctr=08030 block=08029 update=1 done=0 cmd(vx,wz)=(+0.000, -0.374) base(vx,wz)=(+0.100, -0.633)
[focus_env=00] step=08029 high_ctr=08031 block=08030 update=1 done=0 cmd(vx,wz)=(+0.000, -0.224) base(vx,wz)=(+0.096, -0.546)
[focus_env=00] step=08030 high_ctr=08032 block=08031 update=1 done=0 cmd(vx,wz)=(+0.000, -0.374) base(vx,wz)=(+0.092, -0.468)
[focus_env=00] step=08031 high_ctr=08033 block=08032 update=1 done=0 cmd(vx,wz)=(+0.000, -0.250) base(vx,wz)=(+0.090, -0.300)
[focus_env=00] step=08032 high_ctr=08034 block=08033 update=1 done=0 cmd(vx,wz)=(+0.000, -0.400) base(vx,wz)=(+0.091, -0.308)
[focus_env=00] step=08033 high_ctr=08035 block=08034 update=1 done=0 cmd(vx,wz)=(+0.000, -0.250) base(vx,wz)=(+0.094, -0.181)
[focus_env=00] step=08034 high_ctr=08036 block=08035 update=1 done=0 cmd(vx,wz)=(+0.000, -0.400) base(vx,wz)=(+0.100, -0.175)
[focus_env=00] step=08035 high_ctr=08037 block=08036 update=1 done=0 cmd(vx,wz)=(+0.000, -0.259) base(vx,wz)=(+0.104, -0.035)
[focus_env=00] step=08036 high_ctr=08038 block=08037 update=1 done=0 cmd(vx,wz)=(+0.000, -0.409) base(vx,wz)=(+0.107, -0.138)
[focus_env=00] step=08037 high_ctr=08039 block=08038 update=1 done=0 cmd(vx,wz)=(+0.000, -0.259) base(vx,wz)=(+0.108, -0.051)
[focus_env=00] step=08038 high_ctr=08040 block=08039 update=1 done=0 cmd(vx,wz)=(+0.000, -0.409) base(vx,wz)=(+0.107, -0.223)
[focus_env=00] step=08039 high_ctr=08041 block=08040 update=1 done=0 cmd(vx,wz)=(+0.000, -0.265) base(vx,wz)=(+0.105, -0.212)
[focus_env=00] step=08040 high_ctr=08042 block=08041 update=1 done=0 cmd(vx,wz)=(+0.000, -0.415) base(vx,wz)=(+0.103, -0.410)
[focus_env=00] step=08041 high_ctr=08043 block=08042 update=1 done=0 cmd(vx,wz)=(+0.000, -0.271) base(vx,wz)=(+0.100, -0.314)
[focus_env=00] step=08042 high_ctr=08044 block=08043 update=1 done=0 cmd(vx,wz)=(+0.000, -0.421) base(vx,wz)=(+0.100, -0.263)
[focus_env=00] step=08043 high_ctr=08045 block=08044 update=1 done=0 cmd(vx,wz)=(+0.000, -0.278) base(vx,wz)=(+0.102, -0.082)
[focus_env=00] step=08044 high_ctr=08046 block=08045 update=1 done=0 cmd(vx,wz)=(+0.000, -0.428) base(vx,wz)=(+0.107, -0.005)
[focus_env=00] step=08045 high_ctr=08047 block=08046 update=1 done=0 cmd(vx,wz)=(+0.000, -0.278) base(vx,wz)=(+0.114, +0.077)
[focus_env=00] step=08046 high_ctr=08048 block=08047 update=1 done=0 cmd(vx,wz)=(+0.000, -0.408) base(vx,wz)=(+0.123, -0.158)
[focus_env=00] step=08047 high_ctr=08049 block=08048 update=1 done=0 cmd(vx,wz)=(+0.000, -0.292) base(vx,wz)=(+0.123, -0.167)
[focus_env=00] step=08048 high_ctr=08050 block=08049 update=1 done=0 cmd(vx,wz)=(+0.000, -0.433) base(vx,wz)=(+0.127, +0.021)
[focus_env=00] step=08049 high_ctr=08051 block=08050 update=1 done=0 cmd(vx,wz)=(+0.000, -0.283) base(vx,wz)=(+0.132, +0.261)
[focus_env=00] step=08050 high_ctr=08052 block=08051 update=1 done=0 cmd(vx,wz)=(+0.000, -0.335) base(vx,wz)=(+0.137, +0.193)
[focus_env=00] step=08051 high_ctr=08053 block=08052 update=1 done=0 cmd(vx,wz)=(+0.000, -0.281) base(vx,wz)=(+0.143, +0.188)
[focus_env=00] step=08052 high_ctr=08054 block=08053 update=1 done=0 cmd(vx,wz)=(+0.000, -0.338) base(vx,wz)=(+0.153, +0.139)
[focus_env=00] step=08053 high_ctr=08055 block=08054 update=1 done=0 cmd(vx,wz)=(+0.000, -0.275) base(vx,wz)=(+0.168, +0.257)
[focus_env=00] step=08054 high_ctr=08056 block=08055 update=1 done=0 cmd(vx,wz)=(+0.000, -0.325) base(vx,wz)=(+0.188, +0.197)
[focus_env=00] step=08055 high_ctr=08057 block=08056 update=1 done=0 cmd(vx,wz)=(+0.000, -0.275) base(vx,wz)=(+0.210, +0.241)
[focus_env=00] step=08056 high_ctr=08058 block=08057 update=1 done=0 cmd(vx,wz)=(+0.000, -0.322) base(vx,wz)=(+0.236, +0.189)
[focus_env=00] step=08057 high_ctr=08059 block=08058 update=1 done=0 cmd(vx,wz)=(+0.000, -0.273) base(vx,wz)=(+0.262, +0.233)
[focus_env=00] step=08058 high_ctr=08060 block=08059 update=1 done=0 cmd(vx,wz)=(+0.000, -0.320) base(vx,wz)=(+0.285, +0.134)
[focus_env=00] step=08059 high_ctr=08061 block=08060 update=1 done=0 cmd(vx,wz)=(+0.000, -0.274) base(vx,wz)=(+0.307, +0.140)
[focus_env=00] step=08060 high_ctr=08062 block=08061 update=1 done=0 cmd(vx,wz)=(+0.000, -0.319) base(vx,wz)=(+0.328, +0.071)
[focus_env=00] step=08061 high_ctr=08063 block=08062 update=1 done=0 cmd(vx,wz)=(+0.000, -0.273) base(vx,wz)=(+0.350, +0.066)
[focus_env=00] step=08062 high_ctr=08064 block=08063 update=1 done=0 cmd(vx,wz)=(+0.000, -0.316) base(vx,wz)=(+0.374, -0.022)
[focus_env=00] step=08063 high_ctr=08065 block=08064 update=1 done=0 cmd(vx,wz)=(+0.000, -0.274) base(vx,wz)=(+0.400, -0.065)
[focus_env=00] step=08064 high_ctr=08066 block=08065 update=1 done=0 cmd(vx,wz)=(+0.000, -0.313) base(vx,wz)=(+0.428, -0.117)
[focus_env=00] step=08065 high_ctr=08067 block=08066 update=1 done=0 cmd(vx,wz)=(+0.000, -0.276) base(vx,wz)=(+0.456, -0.176)
[focus_env=00] step=08066 high_ctr=08068 block=08067 update=1 done=0 cmd(vx,wz)=(+0.000, -0.311) base(vx,wz)=(+0.483, -0.373)
[focus_env=00] step=08067 high_ctr=08069 block=08068 update=1 done=0 cmd(vx,wz)=(+0.000, -0.288) base(vx,wz)=(+0.507, -0.719)
[focus_env=00] step=08068 high_ctr=08070 block=08069 update=1 done=0 cmd(vx,wz)=(+0.000, -0.318) base(vx,wz)=(+0.524, -0.656)
[focus_env=00] step=08069 high_ctr=08071 block=08070 update=1 done=0 cmd(vx,wz)=(+0.000, -0.297) base(vx,wz)=(+0.556, -0.151)
[focus_env=00] step=08070 high_ctr=08072 block=08071 update=1 done=0 cmd(vx,wz)=(+0.000, -0.308) base(vx,wz)=(+0.577, -0.357)
[focus_env=00] step=08071 high_ctr=08073 block=08072 update=1 done=0 cmd(vx,wz)=(+0.000, -0.306) base(vx,wz)=(+0.593, -0.083)
[focus_env=00] step=08072 high_ctr=08074 block=08073 update=1 done=0 cmd(vx,wz)=(+0.000, -0.311) base(vx,wz)=(+0.623, -0.019)
[focus_env=00] step=08073 high_ctr=08075 block=08074 update=1 done=0 cmd(vx,wz)=(+0.000, -0.314) base(vx,wz)=(+0.665, -0.296)
[focus_env=00] step=08074 high_ctr=08076 block=08075 update=1 done=0 cmd(vx,wz)=(+0.000, -0.313) base(vx,wz)=(+0.684, -1.226)
[focus_env=00] step=08075 high_ctr=08077 block=08076 update=1 done=0 cmd(vx,wz)=(+0.000, -0.321) base(vx,wz)=(+0.700, -1.978)
[focus_env=00] step=08076 high_ctr=08078 block=08077 update=1 done=0 cmd(vx,wz)=(+0.000, -0.317) base(vx,wz)=(+0.732, -2.292)
[focus_env=00] step=08077 high_ctr=08079 block=08078 update=1 done=0 cmd(vx,wz)=(+0.000, -0.318) base(vx,wz)=(+0.748, -0.605)
[focus_env=00] step=08078 high_ctr=08080 block=08079 update=1 done=0 cmd(vx,wz)=(+0.000, -0.324) base(vx,wz)=(+0.795, -0.587)
[focus_env=00] step=08079 high_ctr=08081 block=08080 update=1 done=0 cmd(vx,wz)=(+0.000, -0.321) base(vx,wz)=(+0.828, -0.629)
[focus_env=00] step=08080 high_ctr=08082 block=08081 update=1 done=0 cmd(vx,wz)=(+0.000, -0.320) base(vx,wz)=(+0.856, -0.154)
[focus_env=00] step=08081 high_ctr=08083 block=08082 update=1 done=0 cmd(vx,wz)=(+0.000, -0.320) base(vx,wz)=(+0.914, +0.308)
[focus_env=00] step=08082 high_ctr=08084 block=08083 update=1 done=0 cmd(vx,wz)=(+0.000, -0.316) base(vx,wz)=(+0.988, +0.542)
[focus_env=00] step=08083 high_ctr=08085 block=08084 update=1 done=0 cmd(vx,wz)=(+0.000, -0.309) base(vx,wz)=(+1.055, -0.333)
[focus_env=00] step=08084 high_ctr=08086 block=08085 update=1 done=0 cmd(vx,wz)=(+0.000, -0.302) base(vx,wz)=(+1.118, -1.106)
[focus_env=00] step=08085 high_ctr=08087 block=08086 update=1 done=0 cmd(vx,wz)=(+0.000, -0.294) base(vx,wz)=(+1.174, -1.420)
[focus_env=00] step=08086 high_ctr=08088 block=08087 update=1 done=0 cmd(vx,wz)=(+0.000, -0.287) base(vx,wz)=(+1.268, +0.729)
[focus_env=00] step=08087 high_ctr=08089 block=08088 update=1 done=0 cmd(vx,wz)=(+0.000, -0.280) base(vx,wz)=(+1.353, +0.983)
[focus_env=00] step=08088 high_ctr=08090 block=08089 update=1 done=0 cmd(vx,wz)=(+0.000, -0.277) base(vx,wz)=(+1.438, +1.114)
[focus_env=00] step=08089 high_ctr=08091 block=08090 update=1 done=0 cmd(vx,wz)=(+0.000, -0.276) base(vx,wz)=(+1.537, +1.472)
[focus_env=00] step=08090 high_ctr=08092 block=08091 update=1 done=0 cmd(vx,wz)=(+0.000, -0.276) base(vx,wz)=(+1.652, +1.196)
[focus_env=00] step=08091 high_ctr=08093 block=08092 update=1 done=0 cmd(vx,wz)=(+0.000, -0.277) base(vx,wz)=(+1.765, +0.692)
[focus_env=00] step=08092 high_ctr=08094 block=08093 update=1 done=0 cmd(vx,wz)=(+0.000, -0.277) base(vx,wz)=(+1.879, +0.238)
[focus_env=00] step=08093 high_ctr=08095 block=08094 update=1 done=0 cmd(vx,wz)=(+0.000, -0.278) base(vx,wz)=(+1.996, +0.011)
[focus_env=00] step=08094 high_ctr=08096 block=08095 update=1 done=0 cmd(vx,wz)=(+0.000, -0.277) base(vx,wz)=(+2.130, +1.019)
[focus_env=00] step=08095 high_ctr=08097 block=08096 update=1 done=0 cmd(vx,wz)=(+0.000, -0.277) base(vx,wz)=(+2.281, +1.992)
[focus_env=00] step=08096 high_ctr=08098 block=08097 update=1 done=0 cmd(vx,wz)=(+0.000, -0.278) base(vx,wz)=(+2.463, +2.541)
[focus_env=00] step=08097 high_ctr=08099 block=08098 update=1 done=0 cmd(vx,wz)=(+0.000, -0.280) base(vx,wz)=(+2.660, +2.899)
[focus_env=00] step=08098 high_ctr=08100 block=08099 update=1 done=1 cmd(vx,wz)=(+0.000, +0.000) base(vx,wz)=(+2.862, +2.519)
"""
        return load_cmd_trace_from_text(preset_text)
    raise ValueError(f"Unknown preset: {name}")


def apply_eval_overrides(env_cfg, num_envs_cap):
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max(1, num_envs_cap))
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


def replay_lowlevel_commands(replay_args, base_args):
    if base_args.task != "g1":
        raise ValueError(f"Use low-level task --task g1 for replay, got: {base_args.task}")

    trace = load_cmd_trace_preset(replay_args.preset)
    trace_src = f"preset:{replay_args.preset}"

    env_cfg, train_cfg = task_registry.get_cfgs(name=base_args.task)
    apply_eval_overrides(env_cfg, replay_args.num_envs_cap)
    env, _ = task_registry.make_env(name=base_args.task, args=base_args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=base_args.task,
        args=base_args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)

    focus_env_id = min(max(0, int(replay_args.focus_env_id)), env.num_envs - 1)
    print(f"Loaded {len(trace)} cmd steps from: {trace_src}")
    print(f"Replay env.num_envs={env.num_envs} env.dt={env.dt:.4f}s focus_env={focus_env_id}")

    first_done_step = None
    total_step = 0
    seq_idx = 0
    cycle = 0
    while True:
        if replay_args.max_steps > 0 and total_step >= replay_args.max_steps:
            break

        vx_cmd, wz_cmd = trace[seq_idx]

        env.commands[:, 0] = vx_cmd
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = wz_cmd
        env.commands[:, 3] = 0.0

        # Keep the same order as low-level play/high-level wrapper:
        # command injection -> observation refresh -> policy inference.
        obs = env.get_observations()
        actions = policy(obs.detach())
        obs, _, rews, dones, _infos = env.step(actions.detach())

        if total_step % max(1, replay_args.print_every) == 0:
            base_vx = float(env.base_lin_vel[focus_env_id, 0].item())
            base_wz = float(env.base_ang_vel[focus_env_id, 2].item())
            done_focus = int(bool(dones[focus_env_id].item()))
            print(
                f"[replay] total={total_step:06d} cycle={cycle:04d} seq={seq_idx:03d}/{len(trace)-1:03d} done={done_focus} "
                f"cmd(vx,wz)=({vx_cmd:+.3f}, {wz_cmd:+.3f}) "
                f"base(vx,wz)=({base_vx:+.3f}, {base_wz:+.3f}) "
                f"rew_mean={rews.mean().item():+.4f}"
            )

        if first_done_step is None and torch.any(dones):
            first_done_step = total_step

        done_focus = bool(dones[focus_env_id].item())
        should_reset_on_done = done_focus
        should_reset_on_end = (seq_idx + 1 >= len(trace))
        if should_reset_on_done or should_reset_on_end:
            reset_ids = torch.tensor([focus_env_id], dtype=torch.long, device=env.device)
            env.reset_idx(reset_ids)
            obs = env.get_observations()
            seq_idx = 0
            cycle += 1
        else:
            seq_idx += 1

        total_step += 1

    print("=== replay summary ===")
    if first_done_step is None:
        print("first_done_step=None (no done in replay window)")
    else:
        print(f"first_done_step={first_done_step}")


if __name__ == "__main__":
    replay_args, base_args = parse_replay_args()
    replay_lowlevel_commands(replay_args, base_args)
