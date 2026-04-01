from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1VisionRoughCfg(G1RoughCfg):
    """
    Isolated config namespace for vision high-level work.
    Low-level locomotion settings remain compatible with pretrained G1 policy.
    """

    class env(G1RoughCfg.env):
        num_envs = 1
        test = True
        episode_length_s = 60.0

    class terrain(G1RoughCfg.terrain):
        num_rows = 5
        num_cols = 5
        curriculum = False

    class noise(G1RoughCfg.noise):
        add_noise = True

    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction = True
        push_robots = False

    class commands(G1RoughCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 999999.0
        num_commands = 4

        class ranges(G1RoughCfg.commands.ranges):
            # Commands are injected externally by scripts.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]


class G1VisionRoughCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        # Keep low-level checkpoint compatibility with existing g1 runs.
        experiment_name = "g1"

