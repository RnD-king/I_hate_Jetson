import isaacgym

import os
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry


class CommandScheduler:
    """
    테스트용 속도 명령 생성기.

    commands 형식:
        [vx, vy, wz]

    현재 의도:
        - vy는 0 고정
        - vx, wz만 테스트
    """

    def __init__(self, num_envs, device, dt):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        # 최종적으로 env.commands[:, :3]에 넣을 버퍼
        self.commands = torch.zeros((num_envs, 3), device=device, dtype=torch.float)

        # ===== 일반 모드용 파라미터 =====
        self.vx_min = -0.3
        self.vx_max = 1.8

        self.wz_min = -0.8
        self.wz_max = 0.8

        self.hold_steps = 30

        # random_hold용 내부 저장값
        self.shared_vx = 0.0
        self.shared_wz = 0.0

        self.per_env_vx = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.per_env_wz = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # fixed_split_lr용 파라미터
        self.split_vx = 0.3
        self.split_wz = 0.8

    def _set_all_envs_same_command(self, vx, wz):
        self.commands[:, 0] = vx
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = wz
        return self.commands

    def _set_per_env_command(self, vx_tensor, wz_tensor):
        self.commands[:, 0] = vx_tensor
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = wz_tensor
        return self.commands

    def fixed_command(self, vx=0.4, wz=0.0):
        """
        모든 env에 완전히 같은 고정 명령.
        """
        return self._set_all_envs_same_command(vx=vx, wz=wz)

    def random_hold_same_command(self, step):
        """
        모든 env에 같은 랜덤 명령을 주되,
        hold_steps 동안 유지.
        """
        if step % self.hold_steps == 0:
            self.shared_vx = np.random.uniform(self.vx_min, self.vx_max)
            self.shared_wz = np.random.uniform(self.wz_min, self.wz_max)

        return self._set_all_envs_same_command(vx=self.shared_vx, wz=self.shared_wz)

    def random_hold_per_env_command(self, step):
        """
        각 env마다 서로 다른 랜덤 명령을 주되,
        hold_steps 동안 유지.
        """
        if step % self.hold_steps == 0:
            self.per_env_vx = torch.empty(self.num_envs, device=self.device).uniform_(self.vx_min, self.vx_max)
            self.per_env_wz = torch.empty(self.num_envs, device=self.device).uniform_(self.wz_min, self.wz_max)

        return self._set_per_env_command(
            vx_tensor=self.per_env_vx,
            wz_tensor=self.per_env_wz
        )

    def step_sequence_command(self, step):
        """
        시간 구간별로 명령을 바꿔가며 급변 상황을 확인하는 모드.
        모든 env에 같은 명령을 줌.
        """
        t = step * self.dt

        if t < 2.0:
            vx, wz = 0.0, 0.0
        elif t < 4.0:
            vx, wz = 0.3, 0.0
        elif t < 6.0:
            vx, wz = 0.6, 0.0
        elif t < 8.0:
            vx, wz = 0.0, 0.4
        elif t < 10.0:
            vx, wz = 0.4, -0.3
        elif t < 12.0:
            vx, wz = -0.2, 0.0
        else:
            vx, wz = 0.0, 0.0

        return self._set_all_envs_same_command(vx=vx, wz=wz)

    def fixed_split_lr_command(self, vx=None, wz=None):
        """
        env 절반은 +wz, 나머지 절반은 -wz 로 줌.
        좌/우 회전 대칭성 확인용.

        예:
            env 4개면
            0,1 -> +wz
            2,3 -> -wz
        """
        if vx is None:
            vx = self.split_vx
        if wz is None:
            wz = self.split_wz

        half = self.num_envs // 2

        self.commands[:, 0] = vx
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = 0.0

        self.commands[:half, 2] = +wz
        self.commands[half:, 2] = -wz

        return self.commands

    def get_commands(self, mode, step):
        if mode == "fixed":
            return self.fixed_command(vx=0.4, wz=0.0)

        elif mode == "random_hold_same":
            return self.random_hold_same_command(step)

        elif mode == "random_hold_per_env":
            return self.random_hold_per_env_command(step)

        elif mode == "step_sequence":
            return self.step_sequence_command(step)

        elif mode == "fixed_split_lr":
            return self.fixed_split_lr_command(vx=self.split_vx, wz=self.split_wz)

        else:
            raise ValueError(f"Unknown mode: {mode}")


def print_scheduler_summary(step, cmds, hold_steps, mode):
    """
    scheduler가 만든 cmds 자체를 출력
    """
    cmd_cpu = cmds.detach().cpu().numpy()

    vx = cmd_cpu[:, 0]
    vy = cmd_cpu[:, 1]
    wz = cmd_cpu[:, 2]

    block_idx = step // hold_steps

    if mode == "fixed_split_lr":
        half = len(vx) // 2
        print(
            f"[step {step:04d} | block {block_idx}] "
            f"SCHED groupA(vx,wz)=({vx[0]:.3f}, {wz[0]:.3f}) | "
            f"groupB(vx,wz)=({vx[half]:.3f}, {wz[half]:.3f})"
        )
    else:
        print(
            f"[step {step:04d} | block {block_idx}] "
            f"SCHED vx(min/mean/max)=({vx.min():.3f}/{vx.mean():.3f}/{vx.max():.3f}) | "
            f"vy(min/mean/max)=({vy.min():.3f}/{vy.mean():.3f}/{vy.max():.3f}) | "
            f"wz(min/mean/max)=({wz.min():.3f}/{wz.mean():.3f}/{wz.max():.3f})"
        )


def print_env_command_summary(step, env_commands, hold_steps, mode, tag):
    """
    env.commands 상태를 출력
    - AFTER_SET
    - AFTER_STEP
    같은 태그로 호출
    """
    cmd_cpu = env_commands.detach().cpu().numpy()

    vx = cmd_cpu[:, 0]
    vy = cmd_cpu[:, 1]
    wz = cmd_cpu[:, 2]

    block_idx = step // hold_steps

    if mode == "fixed_split_lr":
        half = len(vx) // 2
        print(
            f"[step {step:04d} | block {block_idx}] "
            f"{tag} groupA(vx,wz)=({vx[0]:.3f}, {wz[0]:.3f}) | "
            f"groupB(vx,wz)=({vx[half]:.3f}, {wz[half]:.3f})"
        )
    else:
        print(
            f"[step {step:04d} | block {block_idx}] "
            f"{tag} vx(min/mean/max)=({vx.min():.3f}/{vx.mean():.3f}/{vx.max():.3f}) | "
            f"vy(min/mean/max)=({vy.min():.3f}/{vy.mean():.3f}/{vy.max():.3f}) | "
            f"wz(min/mean/max)=({wz.min():.3f}/{wz.mean():.3f}/{wz.max():.3f})"
        )


def print_single_env_debug(step, cmds, env, idx_a=0, idx_b=2):
    """
    특정 두 env에 대해
    - scheduler cmds
    - env.commands
    를 직접 출력
    fixed_split_lr에서 좌/우 비교하기 좋음
    """
    cmds_cpu = cmds.detach().cpu().numpy()
    env_cmd_cpu = env.commands.detach().cpu().numpy()

    print(
        f"  DEBUG step={step} | "
        f"cmds[{idx_a}]={cmds_cpu[idx_a]} env.commands[{idx_a}]={env_cmd_cpu[idx_a]} | "
        f"cmds[{idx_b}]={cmds_cpu[idx_b]} env.commands[{idx_b}]={env_cmd_cpu[idx_b]}"
    )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ===== 테스트용 override =====
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # ===== 중요: 내부 command resampling이 있다면 최대한 꺼보기 =====
    # base config에 commands 항목이 있을 수 있으니, 있으면 override 시도
    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "resampling_time"):
            env_cfg.commands.resampling_time = 999999.0
        if hasattr(env_cfg.commands, "heading_command"):
            env_cfg.commands.heading_command = False

    # ===== 환경 생성 =====
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    print("env.num_envs =", env.num_envs)
    print("env.dt       =", env.dt)
    print("100 steps    =", 100 * env.dt, "seconds")

    # ===== 정책 로드 =====
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ===== 필요하면 JIT export =====
    EXPORT_POLICY = False
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies"
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    # ===== 커맨드 스케줄러 =====
    scheduler = CommandScheduler(
        num_envs=env.num_envs,
        device=env.device,
        dt=env.dt
    )

    # ===============================
    # 테스트 모드 선택
    # ===============================
    # "fixed"
    # "random_hold_same"
    # "random_hold_per_env"
    # "step_sequence"
    # "fixed_split_lr"
    mode = "fixed_split_lr"

    # ===== 일반 모드용 범위 =====
    scheduler.vx_min = -0.3
    scheduler.vx_max = 1.8
    scheduler.wz_min = -0.8
    scheduler.wz_max = 0.8
    scheduler.hold_steps = 30

    # ===== fixed_split_lr 모드용 설정 =====
    # 좌/우 회전 대칭성 확인용
    scheduler.split_vx = 0.0
    scheduler.split_wz = 2.8

    print("mode         =", mode)
    print("vx range     =", (scheduler.vx_min, scheduler.vx_max))
    print("wz range     =", (scheduler.wz_min, scheduler.wz_max))
    print("hold_steps   =", scheduler.hold_steps)
    print("split_vx     =", scheduler.split_vx)
    print("split_wz     =", scheduler.split_wz)

    # ===== 메인 루프 =====
    total_steps = 10 * int(env.max_episode_length)

    for i in range(total_steps):
        # 현재 step에서 넣을 명령 생성
        cmds = scheduler.get_commands(mode=mode, step=i)

        if i % 50 == 0:
            print_scheduler_summary(
                step=i,
                cmds=cmds,
                hold_steps=scheduler.hold_steps,
                mode=mode
            )

        # 환경 명령 덮어쓰기
        env.commands[:, :3] = cmds
        env.commands[:, 3] = 0.0

        if i % 50 == 0:
            print_env_command_summary(
                step=i,
                env_commands=env.commands,
                hold_steps=scheduler.hold_steps,
                mode=mode,
                tag="AFTER_SET"
            )
            print_single_env_debug(i, cmds, env, idx_a=0, idx_b=min(2, env.num_envs - 1))

        # 중요:
        # 방금 넣은 command가 observation에 반영되도록
        # policy 추론 전에 obs를 다시 계산
        obs = env.get_observations()

        # 정책 추론 및 step 진행
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if i % 50 == 0:
            print_env_command_summary(
                step=i,
                env_commands=env.commands,
                hold_steps=scheduler.hold_steps,
                mode=mode,
                tag="AFTER_STEP"
            )
            print_single_env_debug(i, cmds, env, idx_a=0, idx_b=min(2, env.num_envs - 1))


if __name__ == "__main__":
    args = get_args()
    play(args)