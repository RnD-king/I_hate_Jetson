import os
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry


def export_jit(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # env는 크게 중요하지 않지만 최소 생성은 필요
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg
    )

    export_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "policies"
    )
    os.makedirs(export_dir, exist_ok=True)

    export_policy_as_jit(ppo_runner.alg.actor_critic, export_dir)
    print(f"[INFO] Exported JIT policy to: {export_dir}")


if __name__ == "__main__":
    args = get_args()
    export_jit(args)