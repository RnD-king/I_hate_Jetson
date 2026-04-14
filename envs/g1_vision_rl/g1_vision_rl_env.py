from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticRecurrent

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1_vision.features import (
    BASE_FEATURE_NAMES,
    FeatureHistoryStateStack,
    compute_u_err_ctrl,
    extract_base_features,
)
from legged_gym.envs.g1_vision.g1_vision_config import G1VisionRoughCfg
from legged_gym.envs.g1_vision.g1_vision_env import G1VisionRobot
from legged_gym.envs.g1_vision.highlevel_policy import HighLevelCommandAdapter
from legged_gym.envs.g1_vision.scenarios import sample_episode_scenario
from legged_gym.envs.g1_vision_bc.timing import should_update_highlevel
from legged_gym.envs.g1_vision_rl.reward_utils import compute_path_metrics
from legged_gym.utils.helpers import get_load_path


_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_SCRIPTS_DIR))

from g1_pid_module import DotsSplinePidFollower, get_local_pose_rpy  # noqa: E402


class G1VisionRLEnv(VecEnv):
    """
    High-level PPO env wrapper:
    - external action (vx, wz)
    - frozen low-level recurrent locomotion policy inside env.step()
    - vision feature observation built from DotsSplinePidFollower
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        low_cfg = self._build_low_level_env_cfg(cfg)
        self.base_env = G1VisionRobot(
            cfg=low_cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            sim_device=sim_device,
            headless=headless,
        )
        # Disable low-level env's internal auto-reset. We reset explicitly once per
        # high-level decision step so PPO timing stays at fixed macro frequency.
        self._base_reset_idx_impl = self.base_env.reset_idx
        self.base_env.reset_idx = self._noop_base_reset_idx

        # expose base handles for debug draw compatibility
        self.gym = self.base_env.gym
        self.sim = self.base_env.sim
        self.viewer = self.base_env.viewer
        self.envs = self.base_env.envs
        self.env_origins = self.base_env.env_origins
        self.root_states = self.base_env.root_states
        self.dt = float(self.base_env.dt)
        self.device = self.base_env.device

        self.num_envs = int(self.base_env.num_envs)
        self.num_actions = 2
        self.num_privileged_obs = None

        self.low_steps_per_decision = max(1, int(self.cfg.high_level.hold_steps))
        self.perception_hold_steps = max(
            1, int(getattr(self.cfg.high_level, "perception_hold_steps", self.low_steps_per_decision))
        )
        self.history_state = max(1, int(getattr(self.cfg.high_level, "history_state", 2)))
        self.num_obs = int(len(BASE_FEATURE_NAMES) * self.history_state)
        # One PPO env.step corresponds to one high-level decision period.
        self.max_episode_length = max(
            1, int(round(float(self.cfg.env.episode_length_s) / (self.dt * float(self.low_steps_per_decision))))
        )

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float32)
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras: Dict = {}

        self.follower = DotsSplinePidFollower(
            num_envs=self.num_envs,
            device=self.device,
            env_dt=self.dt,
            seed=int(getattr(self.cfg, "seed", 1)),
        )
        self.follower.hold_steps = int(self.low_steps_per_decision)
        self.follower.command_interp_steps = int(self.cfg.high_level.command_interp_steps)
        self.follower.vx_min = float(self.cfg.high_level.vx_min)
        self.follower.vx_max = float(self.cfg.high_level.vx_max)
        self.follower.wz_min = float(self.cfg.high_level.wz_min)
        self.follower.wz_max = float(self.cfg.high_level.wz_max)

        # ensure one manual spec per env for per-env path reset
        base_spec = copy.deepcopy(self.follower.manual_block_specs[0])
        self.follower.manual_block_specs = [copy.deepcopy(base_spec) for _ in range(self.num_envs)]

        # initialize path caches
        self.follower.setup_random_dotted_spline_paths()

        self.feature_stack = FeatureHistoryStateStack(
            num_envs=self.num_envs,
            feature_dim=len(BASE_FEATURE_NAMES),
            history_state=self.history_state,
            device=self.device,
        )
        self._init_obs_normalization()

        self.command_adapter = HighLevelCommandAdapter(self.follower)
        self.command_mode = str(self.cfg.high_level.command_mode).lower()

        self.target_cmd = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.last_cmd = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.prev_cmd = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)

        self.prev_local_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.prev_path_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.prev_path_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.segment_count = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.prev_segment_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.last_path_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_path_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_heading_err = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.ep_reward_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ep_comp_sums = {
            "rew_progress_ratio": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_forward_progress": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_segment_pass": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_path_dist": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_path_dist_out": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_heading_err": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_cmd_smooth": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_turn_mag": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_wrong_turn": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_recovery_tracking": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_blind_fast": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_blind_turn": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_safe_cruise": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_alive": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_success": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_success_time": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_failure": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_no_progress": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rew_spin_in_place": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rate_recovery_perception": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rate_recovery_tracking": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rate_blind_fast": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rate_blind_turn": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "rate_safe_cruise": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
        }

        self.last_commands_4d = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.last_success_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        seed_base = int(getattr(self.cfg, "seed", 1))
        seed_offset = int(getattr(self.cfg.scenario, "seed_offset", 0))
        self.scenario_rng = np.random.default_rng(seed_base + seed_offset)
        self.scenario_episode_counter = 0

        self.init_y_range = torch.full((self.num_envs,), 0.1, device=self.device, dtype=torch.float32)
        self.init_yaw_range = torch.full((self.num_envs,), 0.2, device=self.device, dtype=torch.float32)

        # low-level step counter used by adapter timing
        self._hl_step_counter = 0
        # PPO (high-level decision) step counter
        self.global_step = 0
        # low-level simulation step counter
        self.global_low_step = 0

        self._init_recovery_split_state()
        self.low_level_actor = self._load_low_level_actor()

        self.reset()

    def _init_recovery_split_state(self):
        rec_cfg = getattr(self.cfg, "recovery", None)
        self.rec_n_visible_low = float(getattr(rec_cfg, "n_visible_low", 2.0))
        self.rec_n_visible_high = float(getattr(rec_cfg, "n_visible_high", 4.0))
        self.rec_tracking_enter_path_dist = float(getattr(rec_cfg, "tracking_enter_path_dist", 0.40))
        self.rec_tracking_exit_path_dist = float(getattr(rec_cfg, "tracking_exit_path_dist", 0.28))
        self.rec_tracking_enter_heading = float(getattr(rec_cfg, "tracking_enter_heading", 0.60))
        self.rec_tracking_exit_heading = float(getattr(rec_cfg, "tracking_exit_heading", 0.40))

        hysteresis_decisions = int(getattr(rec_cfg, "hysteresis_decisions", 1))
        self.rec_hysteresis_steps = max(1, hysteresis_decisions * int(self.low_steps_per_decision))

        self.rec_perception_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.rec_tracking_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.rec_perc_enter_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rec_perc_exit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rec_track_enter_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rec_track_exit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _update_recovery_split_state(self, path_dist: torch.Tensor, heading_err: torch.Tensor):
        n_visible = self.follower.n_visible
        abs_heading = torch.abs(heading_err)

        perc_enter = n_visible <= self.rec_n_visible_low
        perc_exit = n_visible >= self.rec_n_visible_high

        track_enter = (n_visible >= self.rec_n_visible_high) & (
            (path_dist >= self.rec_tracking_enter_path_dist) | (abs_heading >= self.rec_tracking_enter_heading)
        )
        track_exit = (n_visible < self.rec_n_visible_high) | (
            (path_dist <= self.rec_tracking_exit_path_dist) & (abs_heading <= self.rec_tracking_exit_heading)
        )

        self.rec_perc_enter_count = torch.where(perc_enter, self.rec_perc_enter_count + 1, torch.zeros_like(self.rec_perc_enter_count))
        self.rec_perc_exit_count = torch.where(perc_exit, self.rec_perc_exit_count + 1, torch.zeros_like(self.rec_perc_exit_count))
        self.rec_track_enter_count = torch.where(track_enter, self.rec_track_enter_count + 1, torch.zeros_like(self.rec_track_enter_count))
        self.rec_track_exit_count = torch.where(track_exit, self.rec_track_exit_count + 1, torch.zeros_like(self.rec_track_exit_count))

        enter_k = self.rec_hysteresis_steps
        exit_k = self.rec_hysteresis_steps

        self.rec_perception_active = torch.where(
            self.rec_perc_enter_count >= enter_k,
            torch.ones_like(self.rec_perception_active),
            self.rec_perception_active,
        )
        self.rec_perception_active = torch.where(
            self.rec_perc_exit_count >= exit_k,
            torch.zeros_like(self.rec_perception_active),
            self.rec_perception_active,
        )

        self.rec_tracking_active = torch.where(
            self.rec_track_enter_count >= enter_k,
            torch.ones_like(self.rec_tracking_active),
            self.rec_tracking_active,
        )
        self.rec_tracking_active = torch.where(
            self.rec_track_exit_count >= exit_k,
            torch.zeros_like(self.rec_tracking_active),
            self.rec_tracking_active,
        )

        # Prioritize perception-recovery over tracking-recovery.
        self.rec_tracking_active = self.rec_tracking_active & (~self.rec_perception_active)

        return self.rec_perception_active, self.rec_tracking_active

    def _init_obs_normalization(self):
        self.obs_norm_mode = str(getattr(self.cfg.high_level, "obs_norm_mode", "none")).lower()
        self.obs_norm_clip = float(getattr(self.cfg.high_level, "obs_norm_clip", 8.0))

        self.obs_mean = torch.zeros(self.num_obs, device=self.device, dtype=torch.float32)
        self.obs_std = torch.ones(self.num_obs, device=self.device, dtype=torch.float32)
        self.obs_norm_enabled = self.obs_norm_mode != "none"

        if self.obs_norm_mode == "none":
            return

        if self.obs_norm_mode != "bc":
            raise ValueError(f"Unsupported obs_norm_mode: {self.obs_norm_mode}")

        ckpt_path = str(getattr(self.cfg.high_level, "obs_norm_ckpt", "")).strip()
        if len(ckpt_path) == 0:
            raise ValueError("obs_norm_mode='bc' requires cfg.high_level.obs_norm_ckpt")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        if ("feature_mean" not in ckpt) or ("feature_std" not in ckpt):
            raise ValueError(f"BC checkpoint missing feature_mean/std: {ckpt_path}")

        mean_np = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        std_np = np.asarray(ckpt["feature_std"], dtype=np.float32)
        std_np = np.where(std_np < 1e-6, 1.0, std_np)

        if mean_np.shape[0] != self.num_obs:
            raise ValueError(
                f"Obs dim mismatch with BC stats: env num_obs={self.num_obs}, "
                f"bc_stats_dim={mean_np.shape[0]} (history_state mismatch likely)"
            )

        self.obs_mean = torch.tensor(mean_np, device=self.device, dtype=torch.float32)
        self.obs_std = torch.tensor(std_np, device=self.device, dtype=torch.float32)
        print(
            f"[g1_vision_rl] obs norm from BC stats: dim={self.num_obs} "
            f"clip={self.obs_norm_clip:.1f} ckpt={ckpt_path}"
        )

    def _noop_base_reset_idx(self, env_ids):
        # Called from base_env.post_physics_step(). Keep terminal flags but defer
        # actual state reset to wrapper-level reset at macro-step boundary.
        if len(env_ids) == 0:
            return
        if bool(getattr(self.base_env.cfg.env, "send_timeouts", False)):
            self.base_env.extras["time_outs"] = self.base_env.time_out_buf

    def _hard_reset_base_env(self, env_ids):
        self._base_reset_idx_impl(env_ids)

    @staticmethod
    def _build_low_level_env_cfg(cfg):
        low_cfg = G1VisionRoughCfg()

        low_cfg.seed = int(getattr(cfg, "seed", 1))
        low_cfg.env.num_envs = int(cfg.env.num_envs)
        low_cfg.env.test = bool(cfg.env.test)
        low_cfg.env.episode_length_s = float(cfg.env.episode_length_s)

        low_cfg.terrain.num_rows = int(cfg.terrain.num_rows)
        low_cfg.terrain.num_cols = int(cfg.terrain.num_cols)
        low_cfg.terrain.curriculum = bool(cfg.terrain.curriculum)

        low_cfg.noise.add_noise = bool(cfg.noise.add_noise)
        low_cfg.domain_rand.randomize_friction = bool(cfg.domain_rand.randomize_friction)
        low_cfg.domain_rand.push_robots = bool(cfg.domain_rand.push_robots)

        if hasattr(cfg, "commands"):
            if hasattr(cfg.commands, "resampling_time"):
                low_cfg.commands.resampling_time = float(cfg.commands.resampling_time)
            if hasattr(cfg.commands, "heading_command"):
                low_cfg.commands.heading_command = bool(cfg.commands.heading_command)

        return low_cfg

    def _load_low_level_actor(self):
        ll = self.cfg.low_level
        log_root = str(Path(LEGGED_GYM_ROOT_DIR) / "logs" / str(ll.experiment_name))
        load_path = get_load_path(
            root=log_root,
            load_run=str(ll.load_run),
            checkpoint=int(ll.checkpoint),
        )
        print(f"[g1_vision_rl] loading low-level policy from: {load_path}")

        model = ActorCriticRecurrent(
            num_actor_obs=int(ll.actor_obs_dim),
            num_critic_obs=int(ll.critic_obs_dim),
            num_actions=int(ll.action_dim),
            actor_hidden_dims=list(ll.actor_hidden_dims),
            critic_hidden_dims=list(ll.critic_hidden_dims),
            activation=str(ll.activation),
            rnn_type=str(ll.rnn_type),
            rnn_hidden_size=int(ll.rnn_hidden_size),
            rnn_num_layers=int(ll.rnn_num_layers),
            init_noise_std=float(ll.init_noise_std),
        ).to(self.device)

        ckpt = torch.load(load_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _reset_low_level_memory(self, env_ids: torch.Tensor):
        if hasattr(self.low_level_actor, "memory_a") and getattr(self.low_level_actor.memory_a, "hidden_states", None) is not None:
            self.low_level_actor.memory_a.reset(env_ids)
        if hasattr(self.low_level_actor, "memory_c") and getattr(self.low_level_actor.memory_c, "hidden_states", None) is not None:
            self.low_level_actor.memory_c.reset(env_ids)

    def _set_global_scenario_params(self, scenario):
        self.follower.manual_straight_len_jitter = float(scenario.manual_straight_len_jitter)
        self.follower.manual_turn_radius_jitter = float(scenario.manual_turn_radius_jitter)
        self.follower.manual_turn_angle_jitter_deg = float(scenario.manual_turn_angle_jitter_deg)

        self.follower.per_pt_dropout_prob = float(scenario.per_pt_dropout_prob)
        self.follower.burst_dropout_prob = float(scenario.burst_dropout_prob)
        self.follower.pixel_jitter_std = float(scenario.pixel_jitter_std)

        self.follower.straight_wiggle_amp_min = float(
            min(scenario.straight_wiggle_amp_min, scenario.straight_wiggle_amp_max)
        )
        self.follower.straight_wiggle_amp_max = float(
            max(scenario.straight_wiggle_amp_min, scenario.straight_wiggle_amp_max)
        )
        self.follower.straight_wiggle_freq_min = float(
            min(scenario.straight_wiggle_freq_min, scenario.straight_wiggle_freq_max)
        )
        self.follower.straight_wiggle_freq_max = float(
            max(scenario.straight_wiggle_freq_min, scenario.straight_wiggle_freq_max)
        )

        self.follower.turn_wiggle_amp_min = float(min(scenario.turn_wiggle_amp_min, scenario.turn_wiggle_amp_max))
        self.follower.turn_wiggle_amp_max = float(max(scenario.turn_wiggle_amp_min, scenario.turn_wiggle_amp_max))
        self.follower.turn_wiggle_freq_min = float(min(scenario.turn_wiggle_freq_min, scenario.turn_wiggle_freq_max))
        self.follower.turn_wiggle_freq_max = float(max(scenario.turn_wiggle_freq_min, scenario.turn_wiggle_freq_max))

    def _rebuild_paths_for_envs(self, env_ids: torch.Tensor):
        for env_idx_t in env_ids:
            env_idx = int(env_idx_t.item())

            if self.follower.path_mode == "block":
                dense_np = self.follower._generate_block_path_points(env_idx)
                thin_idx = np.linspace(
                    0,
                    dense_np.shape[0] - 1,
                    min(self.follower.num_waypoints, dense_np.shape[0]),
                    dtype=np.int32,
                )
                wp_np = dense_np[thin_idx]
            else:
                wp_np = self.follower._make_waypoints(env_idx)
                dense_np = self.follower._catmull_rom_chain(wp_np, self.follower.samples_per_seg)

            path_xy = torch.tensor(dense_np, dtype=torch.float32, device=self.device)
            diffs = path_xy[1:] - path_xy[:-1]
            seg_len = torch.sqrt(torch.sum(diffs * diffs, dim=1) + 1e-8)

            s = torch.zeros(path_xy.shape[0], device=self.device)
            s[1:] = torch.cumsum(seg_len, dim=0)

            heading, curvature = self.follower._compute_heading_curvature(path_xy, s)
            dashes = self.follower._build_dash_points(path_xy, s)

            self.follower.waypoints[env_idx] = torch.tensor(wp_np, dtype=torch.float32, device=self.device)
            self.follower.path_points[env_idx] = path_xy
            self.follower.path_s[env_idx] = s
            self.follower.path_heading[env_idx] = heading
            self.follower.path_curvature[env_idx] = curvature
            self.follower.dash_points[env_idx] = dashes

    def _sample_and_apply_scenarios(self, env_ids: torch.Tensor):
        first = True
        for env_idx_t in env_ids:
            env_idx = int(env_idx_t.item())
            scenario = sample_episode_scenario(
                rng=self.scenario_rng,
                preset=str(self.cfg.scenario.preset),
                episode_idx=self.scenario_episode_counter,
            )
            self.scenario_episode_counter += 1

            if first:
                self._set_global_scenario_params(scenario)
                first = False

            self.follower.manual_block_specs[env_idx] = list(scenario.path_blocks)
            self.segment_count[env_idx] = float(max(1, len(scenario.path_blocks)))
            self.init_y_range[env_idx] = float(scenario.init_y_range)
            self.init_yaw_range[env_idx] = float(scenario.init_yaw_range)

        self._rebuild_paths_for_envs(env_ids)

    def _perturb_pose(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        num = int(env_ids.numel())
        y_range = self.init_y_range[env_ids]
        yaw_range = self.init_yaw_range[env_ids]

        y_offsets = (2.0 * torch.rand(num, device=self.device) - 1.0) * y_range
        yaw_offsets = (2.0 * torch.rand(num, device=self.device) - 1.0) * yaw_range

        self.base_env.root_states[env_ids, 0] = self.base_env.env_origins[env_ids, 0]
        self.base_env.root_states[env_ids, 1] = self.base_env.env_origins[env_ids, 1] + y_offsets
        self.base_env.root_states[env_ids, 7:13] = 0.0

        q = quat_from_euler_xyz(
            torch.zeros(num, device=self.device),
            torch.zeros(num, device=self.device),
            yaw_offsets,
        )
        self.base_env.root_states[env_ids, 3:7] = q

        env_ids_i32 = env_ids.to(dtype=torch.int32)
        self.base_env.gym.set_actor_root_state_tensor_indexed(
            self.base_env.sim,
            gymtorch.unwrap_tensor(self.base_env.root_states),
            gymtorch.unwrap_tensor(env_ids_i32),
            len(env_ids_i32),
        )
        self.base_env.gym.refresh_actor_root_state_tensor(self.base_env.sim)

    def _build_target_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        target = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        if bool(self.cfg.high_level.tanh_output):
            a = torch.tanh(target)
            vx_mid = 0.5 * (float(self.follower.vx_min) + float(self.follower.vx_max))
            vx_rng = 0.5 * (float(self.follower.vx_max) - float(self.follower.vx_min))
            wz_mid = 0.5 * (float(self.follower.wz_min) + float(self.follower.wz_max))
            wz_rng = 0.5 * (float(self.follower.wz_max) - float(self.follower.wz_min))
            target = torch.stack([vx_mid + vx_rng * a[:, 0], wz_mid + wz_rng * a[:, 1]], dim=1)

        target[:, 0] = torch.clamp(target[:, 0], float(self.follower.vx_min), float(self.follower.vx_max))
        target[:, 1] = torch.clamp(target[:, 1], float(self.follower.wz_min), float(self.follower.wz_max))
        return target

    def _build_direct_commands(self, target_vx_wz: torch.Tensor) -> torch.Tensor:
        vx = torch.clamp(target_vx_wz[:, 0], float(self.follower.vx_min), float(self.follower.vx_max))
        wz = torch.clamp(target_vx_wz[:, 1], float(self.follower.wz_min), float(self.follower.wz_max))

        self.follower.v_cmd_prev[:] = vx
        self.follower.w_cmd_prev[:] = wz
        self.follower.v_cmd_hold[:] = vx
        self.follower.w_cmd_hold[:] = wz
        self.follower.v_cmd_start[:] = vx
        self.follower.w_cmd_start[:] = wz
        self.follower.v_cmd_goal[:] = vx
        self.follower.w_cmd_goal[:] = wz
        self.follower.interp_countdown.zero_()

        commands = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        commands[:, 0] = vx
        commands[:, 1] = 0.0
        commands[:, 2] = wz
        commands[:, 3] = 0.0
        return commands

    def _refresh_base_observations_after_reset(self):
        # G1 observation requires phase tensors; these are normally populated in
        # _post_physics_step_callback(), but we also need valid obs immediately
        # after explicit reset paths.
        period = 0.8
        offset = 0.5
        self.base_env.phase = (self.base_env.episode_length_buf * self.base_env.dt) % period / period
        self.base_env.phase_left = self.base_env.phase
        self.base_env.phase_right = (self.base_env.phase + offset) % 1
        self.base_env.leg_phase = torch.cat(
            [self.base_env.phase_left.unsqueeze(1), self.base_env.phase_right.unsqueeze(1)], dim=-1
        )
        self.base_env.compute_observations()

    def _update_highlevel_obs_and_metrics(self, step_idx: int = 0, force_perception_update: bool = False):
        local_x, local_y, roll, pitch, yaw = get_local_pose_rpy(self.base_env)
        base_z = self.base_env.root_states[:, 2]

        should_update_perception = bool(force_perception_update) or should_update_highlevel(
            step_idx=step_idx, hold_steps=self.perception_hold_steps
        )
        if should_update_perception:
            self.follower.update_perception(
                local_x=local_x,
                local_y=local_y,
                base_z=base_z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                env_origins=self.base_env.env_origins,
            )

            base_features = extract_base_features(self.follower)
            obs_raw = self.feature_stack.update(base_features)
            if self.obs_norm_enabled:
                obs_norm = (obs_raw - self.obs_mean.unsqueeze(0)) / torch.clamp(self.obs_std.unsqueeze(0), min=1e-6)
                if self.obs_norm_clip > 0.0:
                    obs_norm = torch.clamp(obs_norm, -self.obs_norm_clip, self.obs_norm_clip)
                self.obs_buf[:] = obs_norm
            else:
                self.obs_buf[:] = obs_raw

        path_ratio, path_dist, heading_err = compute_path_metrics(
            follower=self.follower,
            local_x=local_x,
            local_y=local_y,
            yaw=yaw,
        )

        self.last_path_ratio[:] = path_ratio
        self.last_path_dist[:] = path_dist
        self.last_heading_err[:] = heading_err
        return local_x, path_ratio, path_dist, heading_err

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._hard_reset_base_env(env_ids)
        self._reset_low_level_memory(env_ids)

        self.follower.reset_env_state(env_ids)
        self.feature_stack.reset(env_ids)

        self.target_cmd[env_ids] = 0.0
        self.last_cmd[env_ids] = 0.0
        self.prev_cmd[env_ids] = 0.0

        self._sample_and_apply_scenarios(env_ids)
        self._perturb_pose(env_ids)

        # refresh low-level obs after manual state perturbation
        self._refresh_base_observations_after_reset()

        self.episode_length_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False
        self.reset_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0
        self.ep_reward_sum[env_ids] = 0.0
        self.rec_perception_active[env_ids] = False
        self.rec_tracking_active[env_ids] = False
        self.rec_perc_enter_count[env_ids] = 0
        self.rec_perc_exit_count[env_ids] = 0
        self.rec_track_enter_count[env_ids] = 0
        self.rec_track_exit_count[env_ids] = 0
        for key in self.ep_comp_sums:
            self.ep_comp_sums[key][env_ids] = 0.0

        local_x, path_ratio, path_dist, _heading_err = self._update_highlevel_obs_and_metrics(
            step_idx=0, force_perception_update=True
        )
        self.prev_local_x[env_ids] = local_x[env_ids]
        self.prev_path_ratio[env_ids] = path_ratio[env_ids]
        self.prev_path_dist[env_ids] = path_dist[env_ids]
        self.prev_segment_idx[env_ids] = torch.floor(path_ratio[env_ids] * self.segment_count[env_ids])

        self.extras = {"time_outs": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}
        return self.obs_buf, self.privileged_obs_buf

    def _reset_done_envs(self, done_ids: torch.Tensor):
        if done_ids.numel() == 0:
            return

        self._hard_reset_base_env(done_ids)
        self._reset_low_level_memory(done_ids)

        self.follower.reset_env_state(done_ids)
        self.feature_stack.reset(done_ids)

        self.target_cmd[done_ids] = 0.0
        self.last_cmd[done_ids] = 0.0
        self.prev_cmd[done_ids] = 0.0

        self._sample_and_apply_scenarios(done_ids)
        self._perturb_pose(done_ids)

        self._refresh_base_observations_after_reset()

        self.episode_length_buf[done_ids] = 0
        self.time_out_buf[done_ids] = False
        self.ep_reward_sum[done_ids] = 0.0
        self.rec_perception_active[done_ids] = False
        self.rec_tracking_active[done_ids] = False
        self.rec_perc_enter_count[done_ids] = 0
        self.rec_perc_exit_count[done_ids] = 0
        self.rec_track_enter_count[done_ids] = 0
        self.rec_track_exit_count[done_ids] = 0
        for key in self.ep_comp_sums:
            self.ep_comp_sums[key][done_ids] = 0.0

        local_x, path_ratio, path_dist, _heading_err = self._update_highlevel_obs_and_metrics(
            step_idx=0, force_perception_update=True
        )
        self.prev_local_x[done_ids] = local_x[done_ids]
        self.prev_path_ratio[done_ids] = path_ratio[done_ids]
        self.prev_path_dist[done_ids] = path_dist[done_ids]
        self.prev_segment_idx[done_ids] = torch.floor(path_ratio[done_ids] * self.segment_count[done_ids])

    def step(self, actions: torch.Tensor):
        self.extras = {}
        # One PPO action is a high-level command target for one decision period (0.2s by default).
        self.follower._update_recovery_state_vision()
        self.target_cmd[:] = self._build_target_from_action(actions.to(self.device))

        reward_acc = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        done_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        timeout_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        success_any = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        fail_any = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        path_ratio = self.last_path_ratio.clone()
        path_dist = self.last_path_dist.clone()
        path_ratio_final = self.last_path_ratio.clone()
        path_dist_final = self.last_path_dist.clone()

        for _ in range(self.low_steps_per_decision):
            active_mask = ~done_mask
            if self.command_mode == "adapter":
                commands = self.command_adapter.step(
                    target_vx_wz=self.target_cmd,
                    step_idx=self._hl_step_counter,
                )
            else:
                commands = self._build_direct_commands(self.target_cmd)
            if torch.any(done_mask):
                commands[done_mask] = 0.0

            self.base_env.commands[:, :] = commands
            self.last_commands_4d[:] = commands
            self.last_cmd[:, 0] = commands[:, 0]
            self.last_cmd[:, 1] = commands[:, 2]

            low_obs = self.base_env.get_observations()
            with torch.no_grad():
                lower_actions = self.low_level_actor.act_inference(low_obs.detach())

            _obs_ll, _priv_ll, _rews_ll, base_dones, base_infos = self.base_env.step(lower_actions.detach())

            self._hl_step_counter += 1
            self.global_low_step += 1

            local_x, path_ratio, path_dist, heading_err = self._update_highlevel_obs_and_metrics(
                step_idx=self._hl_step_counter - 1,
                force_perception_update=False,
            )
            path_ratio_final[active_mask] = path_ratio[active_mask]
            path_dist_final[active_mask] = path_dist[active_mask]
            rec_perception, rec_tracking = self._update_recovery_split_state(path_dist=path_dist, heading_err=heading_err)

            progress_ratio_delta = torch.clamp(path_ratio - self.prev_path_ratio, -0.25, 0.25)
            path_dist_delta = torch.clamp(path_dist - self.prev_path_dist, -0.25, 0.25)
            forward_delta = torch.clamp(local_x - self.prev_local_x, -0.25, 0.25)
            segment_idx = torch.floor(path_ratio * self.segment_count)
            segment_advance = torch.clamp(segment_idx - self.prev_segment_idx, min=0.0)
            cmd_smooth = torch.abs(self.last_cmd[:, 0] - self.prev_cmd[:, 0]) + 0.5 * torch.abs(
                self.last_cmd[:, 1] - self.prev_cmd[:, 1]
            )

            scales = self.cfg.rewards.scales
            comp_progress = float(scales.progress_ratio) * progress_ratio_delta
            comp_forward = float(scales.forward_progress) * forward_delta
            comp_segment = float(scales.segment_pass) * segment_advance
            path_dist_exp_k = float(getattr(self.cfg.rewards, "path_dist_exp_k", 1.8))
            path_dist_exp_k = max(1e-6, path_dist_exp_k)
            # Exponential shaping: near-path behaves close to linear, far off-track grows fast.
            path_penalty = torch.expm1(path_dist_exp_k * torch.clamp(path_dist, min=0.0)) / path_dist_exp_k
            comp_path = float(scales.path_dist) * path_penalty
            path_dist_out_gate = float(getattr(self.cfg.rewards, "path_dist_out_gate", 0.12))
            outward_delta = torch.clamp(path_dist_delta, min=0.0)
            comp_path_out = (
                float(getattr(scales, "path_dist_out", 0.0))
                * outward_delta
                * (path_dist >= path_dist_out_gate).to(torch.float32)
            )
            comp_heading = float(scales.heading_err) * torch.abs(heading_err)
            comp_smooth = float(scales.cmd_smooth) * cmd_smooth
            comp_turn = float(scales.turn_mag) * torch.abs(self.last_cmd[:, 1])
            u_err_ctrl = compute_u_err_ctrl(self.follower)
            w_nom = -self.follower.k_u * u_err_ctrl - self.follower.k_slope * self.follower.vision_slope
            wrong_turn_ref_eps = float(getattr(self.cfg.rewards, "wrong_turn_ref_eps", 0.10))
            wrong_turn = torch.clamp(-self.last_cmd[:, 1] * torch.sign(w_nom), min=0.0)
            wrong_turn = wrong_turn * (torch.abs(w_nom) >= wrong_turn_ref_eps).to(torch.float32)
            comp_wrong_turn = float(getattr(scales, "wrong_turn", 0.0)) * wrong_turn
            comp_recovery_tracking = float(getattr(scales, "recovery_tracking", 0.0)) * rec_tracking.to(torch.float32)
            blind_fast_vx_eps = float(getattr(self.cfg.rewards, "blind_fast_vx_eps", 0.35))
            blind_turn_wz_eps = float(getattr(self.cfg.rewards, "blind_turn_wz_eps", 0.60))
            blind_fast_mask = rec_perception & (self.last_cmd[:, 0] >= blind_fast_vx_eps)
            blind_turn_mask = rec_perception & (torch.abs(self.last_cmd[:, 1]) >= blind_turn_wz_eps)
            comp_blind_fast = float(getattr(scales, "blind_fast", 0.0)) * blind_fast_mask.to(torch.float32)
            comp_blind_turn = float(getattr(scales, "blind_turn", 0.0)) * blind_turn_mask.to(torch.float32)
            safe_cruise_path_dist = float(getattr(self.cfg.rewards, "safe_cruise_path_dist", 0.32))
            safe_cruise_heading = float(getattr(self.cfg.rewards, "safe_cruise_heading", 0.30))
            safe_cruise_vx_floor = float(getattr(self.cfg.rewards, "safe_cruise_vx_floor", 0.45))
            safe_cruise_mask = (
                (~rec_perception)
                & (~rec_tracking)
                & (self.follower.n_visible >= self.rec_n_visible_high)
                & (path_dist <= safe_cruise_path_dist)
                & (torch.abs(heading_err) <= safe_cruise_heading)
            )
            comp_safe_cruise = (
                float(getattr(scales, "safe_cruise", 0.0))
                * safe_cruise_mask.to(torch.float32)
                * (self.last_cmd[:, 0] - safe_cruise_vx_floor)
            )
            comp_alive = torch.full_like(comp_progress, float(scales.alive))

            base_dones = base_dones.to(self.device).bool()
            base_time_outs = self.base_env.time_out_buf.to(self.device).bool()

            term_cfg = self.cfg.termination
            success_mask = (
                (path_ratio >= float(term_cfg.success_progress_ratio))
                & (path_dist <= float(term_cfg.success_path_dist))
            )
            offtrack_mask = path_dist >= float(term_cfg.fail_path_dist)

            done_sub = base_dones | offtrack_mask
            if bool(term_cfg.stop_on_success):
                done_sub = done_sub | success_mask
            done_sub = done_sub & active_mask
            fail_sub = done_sub & (~base_time_outs) & (~success_mask)

            comp_success = float(scales.success) * success_mask.to(torch.float32)
            time_left_ratio = 1.0 - (self.episode_length_buf.to(torch.float32) / float(self.max_episode_length))
            time_left_ratio = torch.clamp(time_left_ratio, 0.0, 1.0)
            comp_success_time = float(scales.success_time) * success_mask.to(torch.float32) * time_left_ratio
            comp_failure = float(scales.failure) * fail_sub.to(torch.float32)
            no_progress_eps = float(getattr(self.cfg.rewards, "no_progress_ratio_eps", 1e-4))
            no_progress_mask = torch.abs(progress_ratio_delta) < no_progress_eps
            comp_no_progress = float(scales.no_progress) * no_progress_mask.to(torch.float32)
            spin_forward_eps = float(getattr(self.cfg.rewards, "spin_forward_eps", 2e-3))
            spin_mask = torch.abs(forward_delta) < spin_forward_eps
            comp_spin = float(scales.spin_in_place) * spin_mask.to(torch.float32) * torch.abs(self.last_cmd[:, 1])

            step_reward = (
                comp_progress
                + comp_forward
                + comp_segment
                + comp_path
                + comp_path_out
                + comp_heading
                + comp_smooth
                + comp_turn
                + comp_wrong_turn
                + comp_recovery_tracking
                + comp_blind_fast
                + comp_blind_turn
                + comp_safe_cruise
                + comp_alive
                + comp_success
                + comp_success_time
                + comp_failure
                + comp_no_progress
                + comp_spin
            )
            step_reward = torch.clamp(step_reward, float(self.cfg.rewards.clip_min), float(self.cfg.rewards.clip_max))

            active_f = active_mask.to(torch.float32)
            reward_acc += step_reward * active_f
            self.ep_reward_sum += step_reward * active_f

            self.ep_comp_sums["rew_progress_ratio"] += comp_progress * active_f
            self.ep_comp_sums["rew_forward_progress"] += comp_forward * active_f
            self.ep_comp_sums["rew_segment_pass"] += comp_segment * active_f
            self.ep_comp_sums["rew_path_dist"] += comp_path * active_f
            self.ep_comp_sums["rew_path_dist_out"] += comp_path_out * active_f
            self.ep_comp_sums["rew_heading_err"] += comp_heading * active_f
            self.ep_comp_sums["rew_cmd_smooth"] += comp_smooth * active_f
            self.ep_comp_sums["rew_turn_mag"] += comp_turn * active_f
            self.ep_comp_sums["rew_wrong_turn"] += comp_wrong_turn * active_f
            self.ep_comp_sums["rew_recovery_tracking"] += comp_recovery_tracking * active_f
            self.ep_comp_sums["rew_blind_fast"] += comp_blind_fast * active_f
            self.ep_comp_sums["rew_blind_turn"] += comp_blind_turn * active_f
            self.ep_comp_sums["rew_safe_cruise"] += comp_safe_cruise * active_f
            self.ep_comp_sums["rew_alive"] += comp_alive * active_f
            self.ep_comp_sums["rew_success"] += comp_success * active_f
            self.ep_comp_sums["rew_success_time"] += comp_success_time * active_f
            self.ep_comp_sums["rew_failure"] += comp_failure * active_f
            self.ep_comp_sums["rew_no_progress"] += comp_no_progress * active_f
            self.ep_comp_sums["rew_spin_in_place"] += comp_spin * active_f
            per_step_to_decision = 1.0 / float(self.low_steps_per_decision)
            self.ep_comp_sums["rate_recovery_perception"] += rec_perception.to(torch.float32) * active_f * per_step_to_decision
            self.ep_comp_sums["rate_recovery_tracking"] += rec_tracking.to(torch.float32) * active_f * per_step_to_decision
            self.ep_comp_sums["rate_blind_fast"] += blind_fast_mask.to(torch.float32) * active_f * per_step_to_decision
            self.ep_comp_sums["rate_blind_turn"] += blind_turn_mask.to(torch.float32) * active_f * per_step_to_decision
            self.ep_comp_sums["rate_safe_cruise"] += safe_cruise_mask.to(torch.float32) * active_f * per_step_to_decision

            done_mask |= done_sub
            timeout_mask |= (base_time_outs & active_mask)
            success_any |= (success_mask & active_mask)
            fail_any |= (fail_sub & active_mask)

            alive = active_mask & (~done_sub)
            self.prev_local_x[alive] = local_x[alive]
            self.prev_path_ratio[alive] = path_ratio[alive]
            self.prev_path_dist[alive] = path_dist[alive]
            self.prev_segment_idx[alive] = segment_idx[alive]
            self.prev_cmd[alive] = self.last_cmd[alive]

        # high-level (decision-step) timeout
        self.episode_length_buf += 1
        self.global_step += 1
        macro_timeouts = (~done_mask) & (self.episode_length_buf >= int(self.max_episode_length))
        done_mask |= macro_timeouts
        timeout_mask |= macro_timeouts

        self.rew_buf[:] = reward_acc
        self.reset_buf[:] = done_mask.to(torch.long)
        self.time_out_buf[:] = timeout_mask
        self.last_success_mask[:] = success_any
        self.last_path_ratio[:] = path_ratio_final
        self.last_path_dist[:] = path_dist_final

        self.extras["time_outs"] = timeout_mask

        done_ids = torch.nonzero(done_mask).flatten()
        if done_ids.numel() > 0:
            done_lens = torch.clamp(self.episode_length_buf[done_ids].to(torch.float32), min=1.0)
            episode_info = {
                "rew_total": torch.mean(self.ep_reward_sum[done_ids]),
                "rew_total_mean": torch.mean(self.ep_reward_sum[done_ids] / done_lens),
                "path_ratio": torch.mean(path_ratio_final[done_ids]),
                "path_dist": torch.mean(path_dist_final[done_ids]),
                "success_rate": torch.mean(success_any[done_ids].to(torch.float32)),
                "fail_rate": torch.mean(fail_any[done_ids].to(torch.float32)),
            }
            for key, tensor in self.ep_comp_sums.items():
                episode_info[key] = torch.mean(tensor[done_ids] / done_lens)
            self.extras["episode"] = episode_info

        self._reset_done_envs(done_ids)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
