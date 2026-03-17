import os

import torch

from rsl_rl.modules import ActorCriticRecurrent

from legged_gym.envs.g1.g1_env import G1Robot


class G1HighLevelEnv(G1Robot):
    """
    High-level env (v0).

    PPO-facing interface:
      - obs: 10D
      - action: 2D [vx_cmd, wz_cmd] in [-1, 1]

    Internally:
      - decode to safe command range
      - apply hold + rate limit
      - send [vx, 0, wz, 0] to low-level G1 walker
      - infer 12D joint actions from pretrained recurrent low-level policy
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.high_cfg = cfg.high_level

        self.high_obs_dim_ = cfg.env.num_observations
        self.high_priv_obs_dim_ = cfg.env.num_privileged_obs
        self.high_act_dim_ = cfg.env.num_actions

        self.low_obs_dim_ = 47
        self.low_priv_obs_dim_ = 50
        self.low_act_dim_ = 12

        # Initialize base G1 with low-level dimensions.
        orig_num_obs = cfg.env.num_observations
        orig_num_priv = cfg.env.num_privileged_obs
        orig_num_actions = cfg.env.num_actions
        cfg.env.num_observations = self.low_obs_dim_
        cfg.env.num_privileged_obs = self.low_priv_obs_dim_
        cfg.env.num_actions = self.low_act_dim_

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Restore external/high-level interface dimensions.
        cfg.env.num_observations = orig_num_obs
        cfg.env.num_privileged_obs = orig_num_priv
        cfg.env.num_actions = orig_num_actions
        self.num_obs = self.high_obs_dim_
        self.num_privileged_obs = self.high_priv_obs_dim_
        self.num_actions = self.high_act_dim_

        self.high_obs_buf = torch.zeros(
            self.num_envs, self.high_obs_dim_, device=self.device, dtype=torch.float
        )
        self.low_obs_buf = torch.zeros(
            self.num_envs, self.low_obs_dim_, device=self.device, dtype=torch.float
        )
        self.low_privileged_obs_buf = torch.zeros(
            self.num_envs, self.low_priv_obs_dim_, device=self.device, dtype=torch.float
        )

        self.prev_high_cmd = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.cmd_hold = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.high_step_counter = 0
        self._in_low_level_rollout = False

        self.gt_ey = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.gt_epsi = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.gt_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.prev_local_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.prev_vx_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.prev_wz_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.last_cmd_delta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.metric_ey_abs_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.metric_ey_abs_max = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.metric_progress_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.metric_cmd_delta_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.metric_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # Required by G1Robot.compute_observations() before first sim step.
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)

        self.low_level_actor_critic = None
        self._load_low_level_policy()

        # Bootstrap low-level observation once from current sim state.
        self._in_low_level_rollout = True
        self.compute_low_level_observations()
        self._in_low_level_rollout = False
        self.compute_high_level_observations()

    def _load_low_level_policy(self):
        ckpt_path = os.path.expanduser(self.high_cfg.low_level_checkpoint_path)
        if not ckpt_path:
            raise RuntimeError(
                "high_level.low_level_checkpoint_path is empty. "
                "Set a valid low-level checkpoint path."
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"low_level_checkpoint_path does not exist: {ckpt_path}")

        actor_critic = ActorCriticRecurrent(
            num_actor_obs=self.low_obs_dim_,
            num_critic_obs=self.low_priv_obs_dim_,
            num_actions=self.low_act_dim_,
            actor_hidden_dims=self.high_cfg.low_level_actor_hidden_dims,
            critic_hidden_dims=self.high_cfg.low_level_critic_hidden_dims,
            activation=self.high_cfg.low_level_activation,
            rnn_type=self.high_cfg.low_level_rnn_type,
            rnn_hidden_size=self.high_cfg.low_level_rnn_hidden_size,
            rnn_num_layers=self.high_cfg.low_level_rnn_num_layers,
            init_noise_std=self.high_cfg.low_level_init_noise_std,
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        actor_critic.eval()
        self.low_level_actor_critic = actor_critic

    def _infer_low_level_actions(self, low_obs):
        if self.low_level_actor_critic is None:
            raise RuntimeError("low_level_actor_critic is not loaded.")
        with torch.no_grad():
            return self.low_level_actor_critic.act_inference(low_obs)

    def _decode_high_actions(self, high_actions):
        a = torch.clamp(high_actions, -1.0, 1.0)
        vx_cmd = 0.5 * (a[:, 0] + 1.0) * (self.high_cfg.vx_max - self.high_cfg.vx_min) + self.high_cfg.vx_min
        wz_cmd = 0.5 * (a[:, 1] + 1.0) * (self.high_cfg.wz_max - self.high_cfg.wz_min) + self.high_cfg.wz_min

        vx_cmd = torch.clamp(
            vx_cmd,
            self.prev_high_cmd[:, 0] - self.high_cfg.dv_max,
            self.prev_high_cmd[:, 0] + self.high_cfg.dv_max,
        )
        wz_cmd = torch.clamp(
            wz_cmd,
            self.prev_high_cmd[:, 1] - self.high_cfg.dw_max,
            self.prev_high_cmd[:, 1] + self.high_cfg.dw_max,
        )
        return vx_cmd, wz_cmd

    def compute_low_level_observations(self):
        G1Robot.compute_observations(self)
        self.low_obs_buf = self.obs_buf.clone()
        self.low_privileged_obs_buf = self.privileged_obs_buf.clone()

    def _get_local_pose(self):
        root_pos = self.root_states[:, 0:3]
        local_pos = root_pos - self.env_origins

        local_x = local_pos[:, 0]
        local_y = local_pos[:, 1]

        quat = self.base_quat
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return local_x, local_y, yaw

    def _update_gt_and_metrics(self, vx_cmd, wz_cmd):
        local_x, local_y, yaw = self._get_local_pose()
        progress = local_x - self.prev_local_x

        self.gt_ey = local_y
        self.gt_epsi = yaw
        self.gt_progress = progress
        self.prev_local_x = local_x

        cmd_delta = torch.abs(vx_cmd - self.prev_vx_cmd) + torch.abs(wz_cmd - self.prev_wz_cmd)
        self.last_cmd_delta = cmd_delta
        self.metric_ey_abs_sum += torch.abs(self.gt_ey)
        self.metric_ey_abs_max = torch.maximum(self.metric_ey_abs_max, torch.abs(self.gt_ey))
        self.metric_progress_sum += self.gt_progress
        self.metric_cmd_delta_sum += cmd_delta
        self.metric_steps += 1.0

        self.prev_vx_cmd = vx_cmd
        self.prev_wz_cmd = wz_cmd

    def compute_high_level_observations(self):
        self.high_obs_buf.zero_()

        ey_norm = torch.clamp(
            self.gt_ey / max(self.high_cfg.obs_ey_norm_max, 1e-6),
            -1.0,
            1.0,
        )
        epsi_norm = torch.clamp(
            self.gt_epsi / max(self.high_cfg.obs_epsi_norm_max, 1e-6),
            -1.0,
            1.0,
        )
        prog_norm = torch.clamp(
            self.gt_progress / max(self.high_cfg.obs_progress_norm_max, 1e-6),
            -1.0,
            1.0,
        )

        self.high_obs_buf[:, 0] = ey_norm
        self.high_obs_buf[:, 1] = epsi_norm
        self.high_obs_buf[:, 2] = prog_norm

        # v0: keep most features zero; expose previous command in last 2 dims.
        if self.high_cfg.use_prev_cmd_obs and self.high_obs_dim_ >= 10:
            vx_den = max(self.high_cfg.vx_max - self.high_cfg.vx_min, 1e-6)
            wz_den = max(self.high_cfg.wz_max - self.high_cfg.wz_min, 1e-6)

            vx_norm = 2.0 * (self.prev_high_cmd[:, 0] - self.high_cfg.vx_min) / vx_den - 1.0
            wz_norm = 2.0 * (self.prev_high_cmd[:, 1] - self.high_cfg.wz_min) / wz_den - 1.0

            self.high_obs_buf[:, 8] = torch.clamp(vx_norm, -1.0, 1.0)
            self.high_obs_buf[:, 9] = torch.clamp(wz_norm, -1.0, 1.0)

        self.obs_buf = self.high_obs_buf
        self.privileged_obs_buf = self.high_obs_buf

    def compute_observations(self):
        if self._in_low_level_rollout:
            self.compute_low_level_observations()
        else:
            self.compute_high_level_observations()

    def step(self, high_actions):
        if high_actions.shape[-1] != self.high_act_dim_:
            raise ValueError(
                f"Expected high action dim {self.high_act_dim_}, got {high_actions.shape[-1]}"
            )

        high_actions = high_actions.to(self.device)

        if self.high_step_counter % self.high_cfg.hold_steps == 0:
            vx_cmd, wz_cmd = self._decode_high_actions(high_actions)
            self.cmd_hold[:, 0] = vx_cmd
            self.cmd_hold[:, 1] = wz_cmd
            self.prev_high_cmd[:, :] = self.cmd_hold[:, :]

        vx_cmd = self.cmd_hold[:, 0]
        wz_cmd = self.cmd_hold[:, 1]

        base_done_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._in_low_level_rollout = True
        try:
            for _ in range(self.high_cfg.low_level_decimation):
                self.commands[:, 0] = vx_cmd
                self.commands[:, 1] = 0.0
                self.commands[:, 2] = wz_cmd
                self.commands[:, 3] = 0.0

                low_actions = self._infer_low_level_actions(self.low_obs_buf.detach())

                old_num_actions = self.num_actions
                self.num_actions = self.low_act_dim_
                try:
                    G1Robot.step(self, low_actions.detach())
                    base_done_mask |= self.reset_buf.bool()
                finally:
                    self.num_actions = old_num_actions
        finally:
            self._in_low_level_rollout = False

        self._update_gt_and_metrics(vx_cmd=vx_cmd, wz_cmd=wz_cmd)
        self.compute_high_level_observations()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        self.rew_buf = (
            self.high_cfg.reward_k_progress * self.gt_progress
            - self.high_cfg.reward_k_ey * torch.abs(self.gt_ey)
            - self.high_cfg.reward_k_epsi * torch.abs(self.gt_epsi)
            - self.high_cfg.reward_k_smooth * self.last_cmd_delta
            - self.high_cfg.reward_k_w * torch.abs(wz_cmd)
        )

        fail_mask = (torch.abs(self.gt_ey) > self.high_cfg.fail_ey) | (
            torch.abs(self.gt_epsi) > self.high_cfg.fail_epsi
        )
        self.rew_buf[fail_mask] -= self.high_cfg.fail_penalty

        total_done_mask = base_done_mask | fail_mask
        self.reset_buf = total_done_mask.long()

        high_fail_only = fail_mask & (~base_done_mask)
        high_fail_ids = torch.nonzero(high_fail_only, as_tuple=False).flatten()
        if len(high_fail_ids) > 0:
            self.reset_idx(high_fail_ids)

        steps_safe = torch.clamp(self.metric_steps, min=1.0)
        ey_abs_mean = torch.mean(self.metric_ey_abs_sum / steps_safe).item()
        ey_abs_max = torch.mean(self.metric_ey_abs_max).item()
        progress_sum = torch.mean(self.metric_progress_sum).item()
        cmd_delta_mean = torch.mean(self.metric_cmd_delta_sum / steps_safe).item()
        base_done_rate = base_done_mask.float().mean().item()
        high_fail_rate = fail_mask.float().mean().item()
        total_done_rate = total_done_mask.float().mean().item()
        success_rate = 1.0 - total_done_rate

        self.extras = {
            "high_cmd_vx_mean": torch.mean(vx_cmd).item(),
            "high_cmd_wz_mean": torch.mean(wz_cmd).item(),
            "base_done_rate": base_done_rate,
            "high_fail_rate": high_fail_rate,
            "total_done_rate": total_done_rate,
            "episode": {
                "metric_ey_abs_mean": ey_abs_mean,
                "metric_ey_abs_max": ey_abs_max,
                "metric_progress_sum": progress_sum,
                "metric_success_rate": success_rate,
                "metric_cmd_delta_mean": cmd_delta_mean,
                "metric_base_done_rate": base_done_rate,
                "metric_high_fail_rate": high_fail_rate,
                "metric_total_done_rate": total_done_rate,
            },
        }

        self.high_step_counter += 1
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        self.prev_high_cmd[env_ids] = 0.0
        self.cmd_hold[env_ids] = 0.0

        if self.low_level_actor_critic is not None:
            memory_a = self.low_level_actor_critic.memory_a
            if memory_a.hidden_states is not None:
                memory_a.reset(env_ids)

        self.gt_ey[env_ids] = 0.0
        self.gt_epsi[env_ids] = 0.0
        self.gt_progress[env_ids] = 0.0
        self.prev_local_x[env_ids] = 0.0
        self.prev_vx_cmd[env_ids] = 0.0
        self.prev_wz_cmd[env_ids] = 0.0
        self.last_cmd_delta[env_ids] = 0.0

        self.metric_ey_abs_sum[env_ids] = 0.0
        self.metric_ey_abs_max[env_ids] = 0.0
        self.metric_progress_sum[env_ids] = 0.0
        self.metric_cmd_delta_sum[env_ids] = 0.0
        self.metric_steps[env_ids] = 0.0

        prev_rollout_flag = self._in_low_level_rollout
        self._in_low_level_rollout = True
        self.compute_low_level_observations()
        local_x, _, _ = self._get_local_pose()
        self.prev_local_x[env_ids] = local_x[env_ids]
        self._in_low_level_rollout = prev_rollout_flag
        if not prev_rollout_flag:
            self.compute_high_level_observations()
