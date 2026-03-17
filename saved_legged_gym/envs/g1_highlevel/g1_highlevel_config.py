from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1HighLevelCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 10
        num_privileged_obs = 10
        num_actions = 2
        episode_length_s = 20.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 999999.0
        num_commands = 4

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.3, 1.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        push_robots = False
        randomize_base_mass = False

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_yaw": 100,
            "hip_roll": 100,
            "hip_pitch": 100,
            "knee": 150,
            "ankle": 40,
        }
        damping = {
            "hip_yaw": 2,
            "hip_roll": 2,
            "hip_pitch": 2,
            "knee": 4,
            "ankle": 2,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf"
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 5.0
        clip_actions = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0

    class high_level:
        low_level_checkpoint_path = "/home/noh/unitree/unitree_rl_gym/logs/g1/Mar10_01-38-04_/model_10000.pt"
        low_level_decimation = 5

        # upper policy safe envelope
        vx_min = 0.0
        vx_max = 0.8
        wz_min = -0.5
        wz_max = 0.5

        # command stabilization
        hold_steps = 5
        dv_max = 0.15
        dw_max = 0.15

        # low-level recurrent policy architecture (must match checkpoint)
        low_level_actor_hidden_dims = [32]
        low_level_critic_hidden_dims = [32]
        low_level_activation = "elu"
        low_level_rnn_type = "lstm"
        low_level_rnn_hidden_size = 64
        low_level_rnn_num_layers = 1
        low_level_init_noise_std = 0.8

        # v0 observation: zeros + previous command (last two dims)
        use_prev_cmd_obs = True

        # v1 GT/reward
        reward_k_progress = 2.0
        reward_k_ey = 1.0
        reward_k_epsi = 0.5
        reward_k_smooth = 0.05
        reward_k_w = 0.02

        fail_ey = 0.8
        fail_epsi = 1.2
        fail_penalty = 5.0

        # v1 observation normalization
        obs_ey_norm_max = 0.8
        obs_epsi_norm_max = 1.2
        obs_progress_norm_max = 0.2


class G1HighLevelCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [128, 128]
        critic_hidden_dims = [128, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-4

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        run_name = ""
        experiment_name = "g1_highlevel"
        max_iterations = 5000
