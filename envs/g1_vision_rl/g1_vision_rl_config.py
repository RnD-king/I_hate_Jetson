from legged_gym.envs.g1_vision.g1_vision_config import G1VisionRoughCfg, G1VisionRoughCfgPPO


class G1VisionRLCfg(G1VisionRoughCfg):
    """
    PPO high-level control env config.
    - PPO action: (vx, wz)
    - Low-level G1 locomotion policy stays frozen
    - Observation: fixed-length vision feature history
    """

    class env(G1VisionRoughCfg.env):
        num_envs = 256
        num_observations = 16  # 8 base features x history_state(2)
        num_privileged_obs = None
        num_actions = 2
        test = False
        episode_length_s = 60.0
        send_timeouts = True

    class high_level:
        history_state = 2
        hold_steps = 10 # dt = 0.02  >>  holding 0.2s
        perception_hold_steps = 10
        command_interp_steps = 5  # direct면 의미 없
        command_mode = "direct"  # direct | adapter
        vx_min = 0.10
        vx_max = 0.85
        wz_min = -1.20
        wz_max = 1.20
        tanh_output = False
        obs_norm_mode = "bc"  # none | bc
        obs_norm_ckpt = "/home/noh/unitree/unitree_rl_gym/legged_gym/logs/g1_vision_bc/bc_01_03/model_best.pt"
        obs_norm_clip = 8.0

    class scenario:
        preset = "basic"  # basic | mixed | hard | extreme
        seed_offset = 10000

    class termination:
        stop_on_success = True
        success_progress_ratio = 0.94
        success_path_dist = 0.75
        fail_path_dist = 1.80

    class recovery:
        # Recovery cause split (for reward/log only):
        # - perception_recovery: vision reliability degraded
        # - tracking_recovery: off-track while visibility is reliable
        n_visible_low = 2.0
        n_visible_high = 4.0
        tracking_enter_path_dist = 0.40
        tracking_exit_path_dist = 0.28
        tracking_enter_heading = 0.60
        tracking_exit_heading = 0.40
        hysteresis_decisions = 1

    class rewards:
        clip_min = -5.0
        clip_max = 5.0
        # If progress change is below this, treat as "no progress".
        no_progress_ratio_eps = 5e-4
        # If forward motion is below this while turning, treat as in-place spin.
        spin_forward_eps = 4e-3
        # Exponential path-distance shaping: penalty ~ exp(k*d)-1.
        # Larger k punishes off-track distance more aggressively.
        path_dist_exp_k = 1.5
        # If path distance is already non-trivial, penalize moving farther out.
        path_dist_out_gate = 0.12
        # Ignore tiny nominal turn references to avoid noisy wrong-turn penalties.
        wrong_turn_ref_eps = 0.10
        # In low visibility, penalize aggressive forward speed / turn.
        blind_fast_vx_eps = 0.35
        blind_turn_wz_eps = 0.60
        # When the robot is centered and well-aligned, encourage a minimum cruise speed.
        safe_cruise_vx_floor = 0.45
        safe_cruise_path_dist = 0.32
        safe_cruise_heading = 0.30

        class scales:
            # 전진 보상
            progress_ratio = 30.0
            forward_progress = 2.0
            segment_pass = 3.0
            # 오차 패널티
            path_dist = -0.08
            path_dist_out = -0.16
            heading_err = -0.03
            # 명령 변화 패널티
            cmd_smooth = -0.14
            turn_mag = -0.002
            wrong_turn = -0.06
            recovery_tracking = -0.08
            # 안 보일 때 빠르면 패널티
            blind_fast = -0.12
            blind_turn = -0.04
            # 안전하면 빨리 가라 보상
            safe_cruise = 0.08
            # 생존 보상 or 패널티
            alive = 0.0
            # 성공 보상
            success = 400.0
            success_time = 6.0
            # 실패 패널티
            failure = -2.5
            # 잘 안 가면 패널티
            no_progress = -0.03
            spin_in_place = -0.08

    class low_level:
        experiment_name = "g1"
        load_run = "01"
        checkpoint = 10000

        actor_obs_dim = 47
        critic_obs_dim = 50
        action_dim = 12

        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = "elu"

        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1
        init_noise_std = 0.8


class G1VisionRLCfgPPO(G1VisionRoughCfgPPO):
    class policy:
        init_noise_std = 0.05
        actor_hidden_dims = [128, 128]
        critic_hidden_dims = [128, 128]
        activation = "relu"

    class algorithm(G1VisionRoughCfgPPO.algorithm):
        entropy_coef = 5e-4
        learning_rate = 5e-4
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(G1VisionRoughCfgPPO.runner):
        policy_class_name = "ActorCritic"
        max_iterations = 2000
        num_steps_per_env = 48
        save_interval = 50
        run_name = "ppo"
        experiment_name = "g1_vision_rl"
        resume = False
