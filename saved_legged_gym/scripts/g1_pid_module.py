import math
from dataclasses import dataclass
from dataclasses import field

import numpy as np


from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

import torch

# RL 안 쓰고 PD로만 움직이는 코드임
# 이 코드에 PD 제어 모듈과 시각화 도구들이 구현되어 있음
# 터미널에서 실행은 python legged_gym/scripts/g1_pid_play.py --task=g1 --load_run=Mar10_01-38-04_ --checkpoint=10000


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:  # 각도 텐서를 [-pi, pi] 범위로 정규화
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_to_rpy(quat: torch.Tensor):  # 쿼터니언(x,y,z,w)을 roll/pitch/yaw로 변환
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_local_pose_rpy(env):  # env 기준 로컬 위치와 자세(rpy) 추출
    root_pos = env.root_states[:, 0:3]
    local_pos = root_pos - env.env_origins
    local_x = local_pos[:, 0]
    local_y = local_pos[:, 1]

    roll, pitch, yaw = quat_to_rpy(env.base_quat)
    return local_x, local_y, roll, pitch, yaw


def perturb_initial_pose(env, y_range=0.20, yaw_range=0.30):  # 초기에 소환할 때 y/yaw에 랜덤 오차 주기
    root_states = env.root_states.clone()

    y_offsets = torch.empty(env.num_envs, device=env.device).uniform_(-y_range, y_range)
    yaw_offsets = torch.empty(env.num_envs, device=env.device).uniform_(-yaw_range, yaw_range)

    root_states[:, 0] = env.env_origins[:, 0]
    root_states[:, 1] = env.env_origins[:, 1] + y_offsets
    root_states[:, 2] = env.root_states[:, 2]

    q = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
        yaw_offsets,
    )
    root_states[:, 3:7] = q
    root_states[:, 7:13] = 0.0

    env.root_states[:] = root_states
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)


def reset_done_envs(env, done_ids, follower):  # 종료된 env의 루트 상태/팔로워 상태 초기화
    if len(done_ids) == 0:
        return

    root_states = env.root_states.clone()
    y_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.22, 0.22)
    yaw_offsets = torch.empty(len(done_ids), device=env.device).uniform_(-0.35, 0.35)

    root_states[done_ids, 0] = env.env_origins[done_ids, 0]
    root_states[done_ids, 1] = env.env_origins[done_ids, 1] + y_offsets

    q = quat_from_euler_xyz(
        torch.zeros(len(done_ids), device=env.device),
        torch.zeros(len(done_ids), device=env.device),
        yaw_offsets,
    )
    root_states[done_ids, 3:7] = q
    root_states[done_ids, 7:13] = 0.0

    env.root_states[done_ids] = root_states[done_ids]
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)

    follower.reset_env_state(done_ids)


def arrange_envs_along_y(env, y_gap=2.6):  # env들을 y축으로 나란히 재배치
    if env.num_envs != 2:
        return

    env.env_origins[0, 0] = 0.0
    env.env_origins[1, 0] = 0.0
    env.env_origins[0, 1] = -0.5 * y_gap
    env.env_origins[1, 1] = +0.5 * y_gap
    env.env_origins[:, 2] = 0.0

    root_states = env.root_states.clone()
    root_states[:, 0] = env.env_origins[:, 0]
    root_states[:, 1] = env.env_origins[:, 1]
    root_states[:, 7:13] = 0.0

    env.root_states[:] = root_states
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)


def rot_x(a):  # x축 회전행렬 생성
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(a):  # y축 회전행렬 생성
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(a):  # z축 회전행렬 생성
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rpy_to_rot(roll, pitch, yaw):  # roll/pitch/yaw를 회전행렬로 변환
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


@dataclass
class CameraCfg:  # 카메라 파라미터
    width: int = 640 # 해상도
    height: int = 480
    fx: float = 607.0 # 초점거리
    fy: float = 606.0
    cx: float = 325.5
    cy: float = 239.4
    t_base_cam: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.0, 0.50], dtype=np.float32)) # 카메라 좌표 (베이스 기준)
    mount_rpy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.80, 0.0], dtype=np.float32)) # 카메라 자세 (베이스 기준)
    min_depth_x: float = 0.15
    max_depth_x: float = 4.0


@dataclass
class VizThicknessCfg:  # 시각화 파라미터
    # 점
    dash_cross_half_len: float = 0.02
    dash_cross_thickness: float = 0.008
    tracking_cross_half_len: float = 0.018
    tracking_cross_thickness: float = 0.008
    cam_pos_cross_half_len: float = 0.03
    cam_pos_cross_thickness: float = 0.010
    # 카메라 프레임
    cam_frame_width: float = 0.02
    cam_frame_n_lines: int = 7
    # 화살표
    cmd_arrow_body_width: float = 0.02
    cmd_arrow_body_n_lines: int = 7
    cmd_arrow_head_width: float = 0.02
    cmd_arrow_head_n_lines: int = 7


class DotsSplinePidFollower:
    """
    Vision-only upper controller:
    - random dotted points on floor (dash samples from spline centerline)
    - virtual camera projection
    - image-feature-based upper controller outputting (vx, wz)

    Important:
    - perception synthesis uses world geometry
    - control uses only vision-derived features
    """

    def __init__(self, num_envs, device, env_dt, seed=7):  # 제어/지각/시각화 파라미터 및 런타임 상태 초기화
        self.num_envs = num_envs
        self.device = device
        self.env_dt = env_dt
        self.rng = np.random.default_rng(seed)

        self.vx_min = 0.10 # 속도 범위
        self.vx_max = 1.20
        self.wz_min = -1.90
        self.wz_max = 1.90

        self.hold_steps = 10 # n스텝마다 명령 업뎃
        # 참고: sim 상에서 물리엔진 업뎃은 0.005초마다 / 1 스텝은 4 decimation으로, 1스텝은 0.02초마다 업뎃됨
        # 따라서 명령 주기는 0.2초 = 5Hz
        
        self.dv_max = 0.12 # 스텝당 최대 속도 변화량
        self.dw_max = 0.40
        self.command_interp_steps = 5 # 보간할 스텝 수 (1이면 즉시 반영)

        # ----- vision-only controller -----
        self.v_base = 0.85 # 평소 속도

        self.k_u = 3.00   # y편차에 대한 yaw 보정 게인
        self.k_slope = 3.00  # 기울기에 대한 yaw 보정 게인
        self.k_v_u = 0.35  # y편차에 대한 v 보정 게인
        self.k_v_slope = 0.25  # 기울기에 대한 v 보정 게인

        # look-ahead: 가까운 점(u_bottom) + 먼 점(u_lookahead) 가중 혼합
        self.use_lookahead = True
        self.lookahead_delta_v_px = 190.0   # bottom point보다 위쪽(먼 쪽)으로 볼 픽셀 오프셋
        self.lookahead_alpha_normal = 0.70    # normal 모드 alpha (0=near only, 1=look-ahead only)
        self.lookahead_alpha_recovery = 0.20  # recovery 모드 alpha

        # ----- recovery (vision only) -----
        self.recover_enter_nvis = 2.0  # 점이 1개 이하로 보이면 복귀 모드 진입
        self.recover_exit_nvis = 3.0 # 점이 3개 이상으로 보이면 복귀 모드 탈출
        self.recover_enter_u = 0.70 # y편차 절대값이 0.70 이상이면 복귀 모드 진입
        self.recover_exit_u = 0.35 # y편차 절대값이 0.35 이하이면 복귀 모드 탈출

        self.recover_vx = 0.12 # 복귀 모드에서의 x축 속도 고정
        self.recover_wz = 0.75 # 복귀 모드에서의 z축 회전 속도 고정

        # ----- camera/perception -----
        self.cam_cfg = CameraCfg()
        self.use_rp_stabilization = False # 롤/피치 안정화 여부 (실제 카메라는 고정이지만 시점 보정을 할지)
        self.imu_noise_std_deg = 0.0 # imu 노이즈 표준편차 (degrees)

        self.max_centers = 8 # 한 번에 인식 가능한 점 최대 개수 (카메라 시야 내에서)
        self.per_pt_dropout_prob = 0.12 # 각 점마다 독립적으로 사라질 확률 
        self.burst_dropout_prob = 0.04 # 한 프레임 전체가 인식 실패할 확률 
        self.pixel_jitter_std = 5.0 # 점 위치에 픽셀 단위로 가우시안 노이즈 추가 (표준편차 값)

        # ----- waypoint 모드일 때 -----
        self.num_waypoints = 26 # 웨이포인트 개수 (대략 하나당 1.3m)
        self.samples_per_seg = 36 # 웨이포인트 사이를 보간하여 생성할 점 개수 (대략 0.035m 간격)
        self.path_draw_stride = 6 # 시각화할 때 경로 점을 몇 개씩 건너뛸지 (값이 작을수록 더 촘촘히 그려짐)
        self.path_z = 0.002 # 경로 시각화할 때 z 높이 (바닥에서 약간 띄워서 그리면 더 잘 보임)
        
        self.waypoints = None
        self.path_points = None
        self.path_s = None
        self.path_heading = None
        self.path_curvature = None
        self.dash_points = []

        # ----- block 모드일 때 -----
        self.path_mode = "block"

        self.block_ds = 0.20          # 흰색 점선 간격
        self.dash_len = 0.28 # 흰색 점선 길이
        self.gap_len = 0.22 # 흰색 점선 사이 간격

        # ----- 직선 구간 흔들림 -----
        self.straight_wiggle_amp_min = 0.12   # meters
        self.straight_wiggle_amp_max = 0.20   # meters
        self.straight_wiggle_freq_min = 3.0
        self.straight_wiggle_freq_max = 5.0
        self.straight_wiggle_phase_random = True

        # ----- 커브 구간 흔들림(반지름 방향) -----
        self.turn_wiggle_amp_min = 0.04   # meters
        self.turn_wiggle_amp_max = 0.12   # meters
        self.turn_wiggle_freq_min = 2.0
        self.turn_wiggle_freq_max = 5.0
        self.turn_wiggle_phase_random = True

        self.route_scale = 1.0
        self.manual_block_specs = [  # 경로 생성
            [
                ("S", 8 * self.route_scale),
                ("R", 2.0 * self.route_scale, 90.0),
                ("S", 2 * self.route_scale),
                ("R", 2.0 * self.route_scale, 90.0),
                ("S", 6.0 * self.route_scale),
                ("L", 2.0 * self.route_scale, 90.0),
                ("S", 2.0 * self.route_scale),
                ("L", 2.0 * self.route_scale, 90.0),
                ("S", 8.0 * self.route_scale)
            ],
        ]

        # ----- randomness on top of manual specs ----- 전체 구간을 더 더럽게
        self.manual_straight_len_jitter = 0.0 #0.30   # ± meters
        self.manual_turn_radius_jitter = 0.0 #0.18    # ± meters
        self.manual_turn_angle_jitter_deg = 0.0 #6.0  # ± degrees

        # ----- runtime state -----
        self.in_recovery = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.v_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_prev = torch.zeros(self.num_envs, device=self.device)
        self.v_cmd_hold = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_hold = torch.zeros(self.num_envs, device=self.device)
        self.v_cmd_start = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_start = torch.zeros(self.num_envs, device=self.device)
        self.v_cmd_goal = torch.zeros(self.num_envs, device=self.device)
        self.w_cmd_goal = torch.zeros(self.num_envs, device=self.device)
        self.interp_countdown = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.centers_uv = torch.zeros((self.num_envs, self.max_centers, 2), device=self.device)
        self.centers_valid = torch.zeros((self.num_envs, self.max_centers), device=self.device, dtype=torch.bool)
        self.vision_u_err = torch.zeros(self.num_envs, device=self.device)
        self.vision_u_err_la = torch.zeros(self.num_envs, device=self.device)
        self.vision_slope = torch.zeros(self.num_envs, device=self.device)
        self.n_visible = torch.zeros(self.num_envs, device=self.device)

        self.visible_local_points = [torch.zeros((0, 2), device=self.device) for _ in range(self.num_envs)]

        self.debug_cam_pos_w = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.debug_cam_r_wc = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], self.num_envs, axis=0)
        self.viz = VizThicknessCfg()

    # -------------------------------------------------------------------------
    # Path generation
    # -------------------------------------------------------------------------
    def _catmull_rom_chain(self, pts_xy: np.ndarray, samples_per_seg: int):  # 웨이포인트를 스플라인으로 촘촘히 샘플링
        out = []
        k = pts_xy.shape[0]
        for i in range(k - 1):
            p0 = pts_xy[max(i - 1, 0)]
            p1 = pts_xy[i]
            p2 = pts_xy[i + 1]
            p3 = pts_xy[min(i + 2, k - 1)]

            t_values = np.linspace(0.0, 1.0, samples_per_seg, dtype=np.float32)
            for t in t_values:
                t2 = t * t
                t3 = t2 * t
                c = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                out.append(c)
            if i > 0:
                out.pop(-samples_per_seg)
        return np.asarray(out, dtype=np.float32)

    def _compute_heading_curvature(self, path_xy: torch.Tensor, s: torch.Tensor):  # 경로 heading/curvature 계산
        n = path_xy.shape[0]
        prev_idx = torch.clamp(torch.arange(n, device=self.device) - 1, 0, n - 1)
        next_idx = torch.clamp(torch.arange(n, device=self.device) + 1, 0, n - 1)

        tangent = path_xy[next_idx] - path_xy[prev_idx]
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])

        d_heading = wrap_to_pi_torch(heading[next_idx] - heading[prev_idx])
        ds = s[next_idx] - s[prev_idx]
        ds = torch.clamp(ds, min=1e-4)
        curvature = d_heading / ds
        return heading, curvature

    def _make_waypoints(self, env_idx: int):  # 웨이포인트 모드용 기본 경로 생성
        x = np.linspace(0.0, 34.0, self.num_waypoints, dtype=np.float32) # 34m
        y = np.zeros_like(x)

        sign = 1.0 if env_idx < (self.num_envs // 2) else -1.0
        rand_steps = self.rng.uniform(-0.34, 0.34, size=self.num_waypoints - 1)

        for i in range(1, self.num_waypoints):
            wave = 0.12 * sign * math.sin(0.44 * i)
            y[i] = np.clip(y[i - 1] + rand_steps[i - 1] + wave, -2.1, 2.1)

        y[0] = 0.0
        y[-1] *= 0.55
        return np.stack([x, y], axis=1)
    
    def _get_manual_block_specs(self, env_idx: int):  # env 인덱스에 맞는 수동 블록 시퀀스 반환
        """
        Return one manual block spec list for this env.
        If env count > number of spec lists, wrap around.
        """
        return self.manual_block_specs[env_idx % len(self.manual_block_specs)]
    
    def _apply_jitter_to_block_spec(self, block_spec):  # 블록 길이/반경/각도에 랜덤 지터 적용
        """
        block_spec:
          ("S", length)
          ("L", radius, angle_deg)
          ("R", radius, angle_deg)
        """
        kind = block_spec[0]

        if kind == "S":
            base_len = float(block_spec[1])
            length = base_len + self.rng.uniform(
                -self.manual_straight_len_jitter,
                +self.manual_straight_len_jitter,
            )
            length = max(0.6, length)
            return ("S", length)

        elif kind in ("L", "R"):
            base_radius = float(block_spec[1])
            base_angle_deg = float(block_spec[2])

            radius = base_radius + self.rng.uniform(
                -self.manual_turn_radius_jitter,
                +self.manual_turn_radius_jitter,
            )
            radius = max(0.4, radius)

            angle_deg = base_angle_deg + self.rng.uniform(
                -self.manual_turn_angle_jitter_deg,
                +self.manual_turn_angle_jitter_deg,
            )
            angle_deg = max(5.0, angle_deg)

            return (kind, radius, angle_deg)

        else:
            raise ValueError(f"Unknown block type: {kind}")

    def _append_straight_block(self, pts, x, y, heading, length):  # 내부 흔들림 포함 직선 블록 포인트 추가
        """
        Append a mostly-straight segment, but with small lateral wiggle inside the block.
        The start/end of the block stay aligned so block connections remain smooth enough.
        Returns updated (pts, x, y, heading).
        """
        n = max(2, int(length / self.block_ds))
        s_vals = np.linspace(self.block_ds, length, n, dtype=np.float32)

        # forward / normal basis
        fx = math.cos(heading)
        fy = math.sin(heading)
        nx = -math.sin(heading)
        ny =  math.cos(heading)

        # random wiggle parameters for this straight block
        amp = self.rng.uniform(self.straight_wiggle_amp_min, self.straight_wiggle_amp_max)
        freq = self.rng.uniform(self.straight_wiggle_freq_min, self.straight_wiggle_freq_max)
        phase = self.rng.uniform(0.0, 2.0 * math.pi) if self.straight_wiggle_phase_random else 0.0

        # build points
        for s in s_vals:
            t = s / max(length, 1e-6)  # normalized 0~1

            # envelope makes wiggle vanish near both ends
            envelope = math.sin(math.pi * t)

            # internal lateral wiggle
            lateral = amp * envelope * math.sin(2.0 * math.pi * freq * t + phase)

            px = x + s * fx + lateral * nx
            py = y + s * fy + lateral * ny
            pts.append([px, py])

        # keep final endpoint on the nominal straight-line end
        x = x + length * fx
        y = y + length * fy

        return pts, x, y, heading

    def _append_turn_block(self, pts, x, y, heading, radius, turn_angle, direction):  # 원호 회전 블록 포인트 추가
        """
        Append a circular arc.
        direction: +1 for left, -1 for right
        turn_angle: radians, positive magnitude
        Returns updated (pts, x, y, heading).
        """
        # left normal = [-sin(h), cos(h)]
        nx = -math.sin(heading)
        ny =  math.cos(heading)

        # circle center
        cx = x + direction * radius * nx
        cy = y + direction * radius * ny

        # vector from center to current point
        rx0 = x - cx
        ry0 = y - cy
        phi0 = math.atan2(ry0, rx0)

        arc_len = radius * turn_angle
        n = max(3, int(arc_len / self.block_ds))
        alphas = np.linspace(self.block_ds / radius, turn_angle, n, dtype=np.float32)

        # 커브에서도 내부 흔들림을 주되, 시작/끝에서는 0으로 수렴하게 envelope 적용
        amp = self.rng.uniform(self.turn_wiggle_amp_min, self.turn_wiggle_amp_max)
        freq = self.rng.uniform(self.turn_wiggle_freq_min, self.turn_wiggle_freq_max)
        phase = self.rng.uniform(0.0, 2.0 * math.pi) if self.turn_wiggle_phase_random else 0.0

        for a in alphas:
            t = float(a) / max(float(turn_angle), 1e-6)  # normalized 0~1
            envelope = math.sin(math.pi * t)
            radial_offset = amp * envelope * math.sin(2.0 * math.pi * freq * t + phase)
            r_eff = max(0.08, radius + radial_offset)

            phi = phi0 + direction * a
            px = cx + r_eff * math.cos(phi)
            py = cy + r_eff * math.sin(phi)
            pts.append([px, py])

        heading = heading + direction * turn_angle
        heading = math.atan2(math.sin(heading), math.cos(heading))

        x = pts[-1][0]
        y = pts[-1][1]
        return pts, x, y, heading

    def _generate_block_path_points(self, env_idx: int):  # 블록 시퀀스를 따라 전체 경로 포인트 생성
        """
        Generate dense path points directly from manual block specs
        or from random block sequence mode.
        """
        pts = [[0.0, 0.0]]
        x = 0.0
        y = 0.0
        heading = 0.0

        block_specs = self._get_manual_block_specs(env_idx)

        for block_spec in block_specs:
            block_spec_j = self._apply_jitter_to_block_spec(block_spec)
            kind = block_spec_j[0]

            if kind == "S":
                length = float(block_spec_j[1])

                pts, x, y, heading = self._append_straight_block(
                    pts=pts,
                    x=x,
                    y=y,
                    heading=heading,
                    length=length,
                )

            elif kind == "L":
                radius = float(block_spec_j[1])
                angle_deg = float(block_spec_j[2])
                turn_angle = math.radians(angle_deg)

                pts, x, y, heading = self._append_turn_block(
                    pts=pts,
                    x=x,
                    y=y,
                    heading=heading,
                    radius=radius,
                    turn_angle=turn_angle,
                    direction=+1,
                )

            elif kind == "R":
                radius = float(block_spec_j[1])
                angle_deg = float(block_spec_j[2])
                turn_angle = math.radians(angle_deg)

                pts, x, y, heading = self._append_turn_block(
                    pts=pts,
                    x=x,
                    y=y,
                    heading=heading,
                    radius=radius,
                    turn_angle=turn_angle,
                    direction=-1,
                )

            else:
                raise ValueError(f"Unknown block type in manual mode: {kind}")

        pts_np = np.asarray(pts, dtype=np.float32)
        return pts_np

    def _build_dash_points(self, path_xy: torch.Tensor, path_s: torch.Tensor):  # 경로에서 점선(on) 구간 포인트 추출
        cycle = self.dash_len + self.gap_len
        phase = torch.remainder(path_s, cycle)
        mask = phase < self.dash_len
        pts = path_xy[mask]
        return pts

    def setup_random_dotted_spline_paths(self):  # env별 경로/점선/곡률 캐시 초기화
        all_waypoints = []
        all_paths = []
        all_s = []
        all_heading = []
        all_curvature = []
        self.dash_points = []

        for env_idx in range(self.num_envs):
            if self.path_mode == "block":
                dense_np = self._generate_block_path_points(env_idx)
                # block mode has no sparse waypoint skeleton in the old sense,
                # so just keep a thin sampled version for debug compatibility.
                thin_idx = np.linspace(
                    0,
                    dense_np.shape[0] - 1,
                    min(self.num_waypoints, dense_np.shape[0]),
                    dtype=np.int32,
                )
                wp_np = dense_np[thin_idx]
            else:
                wp_np = self._make_waypoints(env_idx)
                dense_np = self._catmull_rom_chain(wp_np, self.samples_per_seg)

            path_xy = torch.tensor(dense_np, dtype=torch.float32, device=self.device)
            diffs = path_xy[1:] - path_xy[:-1]
            seg_len = torch.sqrt(torch.sum(diffs * diffs, dim=1) + 1e-8)

            s = torch.zeros(path_xy.shape[0], device=self.device)
            s[1:] = torch.cumsum(seg_len, dim=0)

            heading, curvature = self._compute_heading_curvature(path_xy, s)
            dashes = self._build_dash_points(path_xy, s)

            all_waypoints.append(torch.tensor(wp_np, dtype=torch.float32, device=self.device))
            all_paths.append(path_xy)
            all_s.append(s)
            all_heading.append(heading)
            all_curvature.append(curvature)
            self.dash_points.append(dashes)

        self.waypoints = all_waypoints
        self.path_points = all_paths
        self.path_s = all_s
        self.path_heading = all_heading
        self.path_curvature = all_curvature

    # -------------------------------------------------------------------------
    # Perception
    # -------------------------------------------------------------------------
    def _camera_pose_world(self, base_pos_w, roll, pitch, yaw):  # 베이스 자세로 월드 카메라 자세 계산
        r_wb = rpy_to_rot(roll, pitch, yaw)

        r_bc = rpy_to_rot(
            float(self.cam_cfg.mount_rpy[0]),
            float(self.cam_cfg.mount_rpy[1]),
            float(self.cam_cfg.mount_rpy[2]),
        )
        r_wc = r_wb @ r_bc
        cam_pos = base_pos_w + (r_wb @ self.cam_cfg.t_base_cam)
        return cam_pos, r_wc

    def _project_world_points(self, world_pts, cam_pos, r_wc):  # 월드 포인트를 카메라 픽셀 좌표로 투영
        rel = world_pts - cam_pos[None, :]
        p_cam = rel @ r_wc

        x = p_cam[:, 0]
        y = p_cam[:, 1]
        z = p_cam[:, 2]

        valid = (x > self.cam_cfg.min_depth_x) & (x < self.cam_cfg.max_depth_x)
        if np.count_nonzero(valid) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        x = x[valid]
        y = y[valid]
        z = z[valid]
        idx_valid = np.nonzero(valid)[0]

        u = self.cam_cfg.cx - self.cam_cfg.fx * (y / x)
        v = self.cam_cfg.cy - self.cam_cfg.fy * (z / x)
        uv = np.stack([u, v], axis=1)

        in_img = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < float(self.cam_cfg.width))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < float(self.cam_cfg.height))
        )
        uv = uv[in_img]
        idx_valid = idx_valid[in_img]
        return uv.astype(np.float32), idx_valid.astype(np.int64)

    def update_perception(self, local_x, local_y, base_z, roll, pitch, yaw, env_origins):  # 가시 점/비전 특징 업데이트
        self.centers_uv.zero_()
        self.centers_valid.zero_()
        self.vision_u_err.zero_()
        self.vision_u_err_la.zero_()
        self.vision_slope.zero_()
        self.n_visible.zero_()

        for i in range(self.num_envs):
            origin = env_origins[i].detach().cpu().numpy()
            dash_local = self.dash_points[i].detach().cpu().numpy()

            dash_world = np.concatenate(
                [
                    origin[0:2][None, :] + dash_local,
                    np.full((dash_local.shape[0], 1), self.path_z, dtype=np.float32),
                ],
                axis=1,
            )

            base_pos_w = np.array(
                [
                    origin[0] + float(local_x[i].item()),
                    origin[1] + float(local_y[i].item()),
                    float(base_z[i].item()),
                ],
                dtype=np.float32,
            )

            cam_pos, r_wc = self._camera_pose_world(
                base_pos_w=base_pos_w,
                roll=float(roll[i].item()),
                pitch=float(pitch[i].item()),
                yaw=float(yaw[i].item()),
            )
            self.debug_cam_pos_w[i] = cam_pos
            self.debug_cam_r_wc[i] = r_wc

            uv, idx_map = self._project_world_points(dash_world, cam_pos, r_wc)
            if uv.shape[0] == 0:
                self.visible_local_points[i] = torch.zeros((0, 2), device=self.device)
                continue

            keep_mask = self.rng.uniform(0.0, 1.0, size=uv.shape[0]) > self.per_pt_dropout_prob
            uv = uv[keep_mask]
            idx_map = idx_map[keep_mask]

            if uv.shape[0] == 0:
                self.visible_local_points[i] = torch.zeros((0, 2), device=self.device)
                continue

            if self.rng.uniform(0.0, 1.0) < self.burst_dropout_prob:
                pick = self.rng.integers(0, uv.shape[0])
                uv = uv[pick:pick + 1]
                idx_map = idx_map[pick:pick + 1]

            uv += self.rng.normal(0.0, self.pixel_jitter_std, size=uv.shape).astype(np.float32)
            uv[:, 0] = np.clip(uv[:, 0], 0.0, float(self.cam_cfg.width - 1))
            uv[:, 1] = np.clip(uv[:, 1], 0.0, float(self.cam_cfg.height - 1))

            order = np.argsort(-uv[:, 1])
            uv = uv[order]
            idx_map = idx_map[order]

            keep_n = min(self.max_centers, uv.shape[0])
            uv = uv[:keep_n]
            idx_map = idx_map[:keep_n]

            self.centers_uv[i, :keep_n] = torch.tensor(uv, device=self.device)
            self.centers_valid[i, :keep_n] = True
            self.n_visible[i] = float(keep_n)

            local_points = self.dash_points[i][torch.tensor(idx_map, device=self.device, dtype=torch.long)]
            self.visible_local_points[i] = local_points

            u_bottom = float(uv[0, 0])
            u_err = (u_bottom - self.cam_cfg.cx) / max(self.cam_cfg.cx, 1.0)
            self.vision_u_err[i] = u_err

            if keep_n >= 2:
                vv = uv[:, 1]
                uu = uv[:, 0]
                a, b = np.polyfit(vv, uu, 1)
                self.vision_slope[i] = float(a) / 120.0

                # look-ahead 지점의 u 오차: bottom보다 위쪽 픽셀의 선형근사값
                v_la = float(uv[0, 1] - self.lookahead_delta_v_px)
                v_la = float(np.clip(v_la, float(np.min(vv)), float(np.max(vv))))
                u_la = float(a * v_la + b)
                u_err_la = (u_la - self.cam_cfg.cx) / max(self.cam_cfg.cx, 1.0)
                self.vision_u_err_la[i] = u_err_la
            else:
                self.vision_slope[i] = 0.0
                self.vision_u_err_la[i] = u_err

    # -------------------------------------------------------------------------
    # Vision-only controller
    # -------------------------------------------------------------------------
    def _update_recovery_state_vision(self):  # 비전 지표 기반 recovery 상태 전이
        enter = (self.n_visible <= self.recover_enter_nvis) | (torch.abs(self.vision_u_err) > self.recover_enter_u)
        exit_ = (self.n_visible >= self.recover_exit_nvis) & (torch.abs(self.vision_u_err) < self.recover_exit_u)

        self.in_recovery = torch.where(enter, torch.ones_like(self.in_recovery), self.in_recovery)
        self.in_recovery = torch.where(exit_, torch.zeros_like(self.in_recovery), self.in_recovery)

    def _clip_and_rate_limit(self, v_raw, w_raw):  # 속도/회전 명령의 범위 및 변화율 제한
        v = torch.clamp(v_raw, self.vx_min, self.vx_max)
        w = torch.clamp(w_raw, self.wz_min, self.wz_max)

        v = torch.clamp(v, self.v_cmd_prev - self.dv_max, self.v_cmd_prev + self.dv_max)
        w = torch.clamp(w, self.w_cmd_prev - self.dw_max, self.w_cmd_prev + self.dw_max)
        return v, w

    def compute_upper_command_from_vision(self, step_idx):  # 비전 기반 상위 명령 계산(보간 포함)
        interp_steps = max(1, int(self.command_interp_steps))

        if step_idx % self.hold_steps == 0:
            self._update_recovery_state_vision()

            u_err_ctrl = self.vision_u_err
            if self.use_lookahead:
                la_alpha_normal = float(np.clip(self.lookahead_alpha_normal, 0.0, 1.0))
                la_alpha_recovery = float(np.clip(self.lookahead_alpha_recovery, 0.0, 1.0))
                la_alpha = torch.where(
                    self.in_recovery,
                    torch.full_like(self.vision_u_err, la_alpha_recovery),
                    torch.full_like(self.vision_u_err, la_alpha_normal),
                )
                u_blend = (1.0 - la_alpha) * self.vision_u_err + la_alpha * self.vision_u_err_la
                enough_pts = self.n_visible >= 2.0
                u_err_ctrl = torch.where(enough_pts, u_blend, self.vision_u_err)

            w_nom = -self.k_u * u_err_ctrl - self.k_slope * self.vision_slope
            v_nom = (
                self.v_base
                - self.k_v_u * torch.abs(u_err_ctrl)
                - self.k_v_slope * torch.abs(self.vision_slope)
            )

            no_vis = self.n_visible < 0.5
            low_vis = self.n_visible < 2.0

            w_nom = torch.where(low_vis, 0.90 * self.w_cmd_prev, w_nom)
            v_nom = torch.where(low_vis, torch.full_like(v_nom, 0.18), v_nom)

            turn_sign = torch.sign(w_nom)
            turn_sign = torch.where(turn_sign == 0.0, torch.sign(self.w_cmd_prev), turn_sign)
            turn_sign = torch.where(turn_sign == 0.0, torch.ones_like(turn_sign), turn_sign)

            recover_v = torch.full_like(v_nom, self.recover_vx)
            recover_w = self.recover_wz * turn_sign

            v_raw = torch.where(self.in_recovery, recover_v, v_nom)
            w_raw = torch.where(self.in_recovery, recover_w, w_nom)

            v_raw = torch.where(no_vis, torch.full_like(v_raw, 0.10), v_raw)
            w_raw = torch.where(no_vis, 0.95 * self.w_cmd_prev, w_raw)

            v_cmd, w_cmd = self._clip_and_rate_limit(v_raw, w_raw)

            self.v_cmd_start = self.v_cmd_hold.clone()
            self.w_cmd_start = self.w_cmd_hold.clone()
            self.v_cmd_goal = v_cmd
            self.w_cmd_goal = w_cmd
            self.interp_countdown[:] = interp_steps

        if interp_steps <= 1:
            self.v_cmd_hold = self.v_cmd_goal
            self.w_cmd_hold = self.w_cmd_goal
            self.interp_countdown.zero_()
        else:
            active = self.interp_countdown > 0
            alpha = (interp_steps - self.interp_countdown + 1).to(torch.float32) / float(interp_steps)
            alpha = torch.clamp(alpha, 0.0, 1.0)

            v_interp = self.v_cmd_start + (self.v_cmd_goal - self.v_cmd_start) * alpha
            w_interp = self.w_cmd_start + (self.w_cmd_goal - self.w_cmd_start) * alpha

            self.v_cmd_hold = torch.where(active, v_interp, self.v_cmd_goal)
            self.w_cmd_hold = torch.where(active, w_interp, self.w_cmd_goal)
            self.interp_countdown = torch.where(active, self.interp_countdown - 1, self.interp_countdown)

        self.v_cmd_prev = self.v_cmd_hold
        self.w_cmd_prev = self.w_cmd_hold

        commands = torch.zeros((self.num_envs, 4), device=self.device)
        commands[:, 0] = self.v_cmd_hold
        commands[:, 1] = 0.0
        commands[:, 2] = self.w_cmd_hold
        commands[:, 3] = 0.0
        return commands

    def reset_env_state(self, done_ids):  # done env의 명령/보간 상태 초기화
        self.in_recovery[done_ids] = False
        self.v_cmd_prev[done_ids] = 0.0
        self.w_cmd_prev[done_ids] = 0.0
        self.v_cmd_hold[done_ids] = 0.0
        self.w_cmd_hold[done_ids] = 0.0
        self.v_cmd_start[done_ids] = 0.0
        self.w_cmd_start[done_ids] = 0.0
        self.v_cmd_goal[done_ids] = 0.0
        self.w_cmd_goal[done_ids] = 0.0
        self.interp_countdown[done_ids] = 0

    # -------------------------------------------------------------------------
    # Debug / print
    # -------------------------------------------------------------------------
    def print_config(self):  # 현재 제어/지각 설정을 콘솔에 출력
        print("dots+spline+vision-only settings:")
        print("  vx range             =", (self.vx_min, self.vx_max))
        print("  wz range             =", (self.wz_min, self.wz_max))
        print("  hold_steps           =", self.hold_steps)
        print("  command_interp_steps =", self.command_interp_steps)
        print("  dv_max               =", self.dv_max)
        print("  dw_max               =", self.dw_max)
        print("  v_base               =", self.v_base)
        print("  k_u                  =", self.k_u)
        print("  k_slope              =", self.k_slope)
        print("  use_lookahead        =", self.use_lookahead)
        print("  lookahead_delta_v_px =", self.lookahead_delta_v_px)
        print("  lookahead_alpha_normal   =", self.lookahead_alpha_normal)
        print("  lookahead_alpha_recovery =", self.lookahead_alpha_recovery)
        print("  k_v_u                =", self.k_v_u)
        print("  k_v_slope            =", self.k_v_slope)
        print("  recover_enter_nvis   =", self.recover_enter_nvis)
        print("  recover_exit_nvis    =", self.recover_exit_nvis)
        print("  recover_enter_u      =", self.recover_enter_u)
        print("  recover_exit_u       =", self.recover_exit_u)
        print("  use_rp_stabilization =", self.use_rp_stabilization)
        print("  per_pt_dropout_prob  =", self.per_pt_dropout_prob)
        print("  burst_dropout_prob   =", self.burst_dropout_prob)
        print("  pixel_jitter_std     =", self.pixel_jitter_std)
        print("  max_centers          =", self.max_centers)


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------
def _draw_cross_thick(env, env_handle, x, y, z, half_len, color, thickness=0.008):  # 십자(+) 마커를 두껍게 그리기
    offs = (0.0, +0.5 * thickness, -0.5 * thickness)

    for dy in offs:
        p1 = np.array([x - half_len, y + dy, z], dtype=np.float32)
        p2 = np.array([x + half_len, y + dy, z], dtype=np.float32)
        env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([p1, p2]).astype(np.float32), color)

    for dx in offs:
        p3 = np.array([x + dx, y - half_len, z], dtype=np.float32)
        p4 = np.array([x + dx, y + half_len, z], dtype=np.float32)
        env.gym.add_lines(env.viewer, env_handle, 1, np.concatenate([p3, p4]).astype(np.float32), color)


def _draw_segment_thick(env, env_handle, p0, p1, color, width=0.01, n_lines=2):  # 선분을 평행 다중선으로 두껍게 그리기
    d = p1 - p0
    dxy = float(np.linalg.norm(d[:2]))
    if dxy < 1e-6:
        return
    nx = -d[1] / dxy
    ny = d[0] / dxy
    offsets = np.linspace(-0.5 * width, 0.5 * width, n_lines, dtype=np.float32)
    for off in offsets:
        shift = np.array([nx * off, ny * off, 0.0], dtype=np.float32)
        a = p0 + shift
        b = p1 + shift
        env.gym.add_lines(
            env.viewer,
            env_handle,
            1,
            np.concatenate([a, b]).astype(np.float32),
            color,
        )


def draw_path_and_dashes(env, follower: DotsSplinePidFollower):  # 바닥 경로 점선(흰 십자) 시각화
    if env.viewer is None:
        return

    env.gym.clear_lines(env.viewer)
    viz = follower.viz

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()
        path = follower.path_points[i].detach().cpu().numpy()

        # for k in range(0, path.shape[0] - 1, follower.path_draw_stride):
        #     p0 = np.array([origin[0] + path[k, 0], origin[1] + path[k, 1], follower.path_z], dtype=np.float32)
        #     p1 = np.array([origin[0] + path[k + 1, 0], origin[1] + path[k + 1, 1], follower.path_z], dtype=np.float32)
        #     env.gym.add_lines(
        #         env.viewer,
        #         env.envs[i],
        #         1,
        #         np.concatenate([p0, p1]).astype(np.float32),
        #         np.array([1.0, 0.0, 0.0], dtype=np.float32),
        #     )

        dashes = follower.dash_points[i].detach().cpu().numpy()
        for p in dashes:
            x = origin[0] + p[0]
            y = origin[1] + p[1]
            z = follower.path_z + 0.01
            _draw_cross_thick(
                env=env,
                env_handle=env.envs[i],
                x=x,
                y=y,
                z=z,
                half_len=viz.dash_cross_half_len,
                color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                thickness=viz.dash_cross_thickness,
            )


def draw_tracking_points(env, follower: DotsSplinePidFollower):  # 현재 추적 중인 가시 포인트(민트 십자) 시각화
    if env.viewer is None:
        return
    viz = follower.viz

    for i in range(env.num_envs):
        origin = env.env_origins[i].detach().cpu().numpy()
        vis = follower.visible_local_points[i]

        if vis.numel() > 0:
            vis_np = vis.detach().cpu().numpy()
            for p in vis_np:
                xv = origin[0] + p[0]
                yv = origin[1] + p[1]
                zv = follower.path_z + 0.04
                _draw_cross_thick(
                    env=env,
                    env_handle=env.envs[i],
                    x=xv,
                    y=yv,
                    z=zv,
                    half_len=viz.tracking_cross_half_len,
                    color=np.array([0.0, 1.0, 1.0], dtype=np.float32),
                    thickness=viz.tracking_cross_thickness,
                )


def _pixel_ray_to_world_dir(u, v, cam_cfg: CameraCfg, r_wc):  # 픽셀 좌표를 월드 광선 방향으로 변환
    x = 1.0
    y = -((u - cam_cfg.cx) / cam_cfg.fx) * x
    z = -((v - cam_cfg.cy) / cam_cfg.fy) * x
    dir_cam = np.array([x, y, z], dtype=np.float32)
    dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-8)
    dir_world = dir_cam @ r_wc.T
    return dir_world


def _intersect_ground(cam_pos, dir_world, z_ground):  # 카메라 광선과 지면의 교차점 계산
    dz = float(dir_world[2])
    if abs(dz) < 1e-6:
        return None
    t = (z_ground - float(cam_pos[2])) / dz
    if t <= 0.0:
        return None
    p = cam_pos + t * dir_world
    return p.astype(np.float32)


def _project_frame_to_ground(cam_pos, r_wc, cam_cfg: CameraCfg, z_ground):  # 카메라 프레임 사각형의 지면 투영점 계산
    w = float(cam_cfg.width - 1)
    h = float(cam_cfg.height - 1)
    cy = float(cam_cfg.cy)

    corner_sets = [
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
        [(0.0, cy), (w, cy), (w, h), (0.0, h)],
        [(0.0, min(cy + 40.0, h)), (w, min(cy + 40.0, h)), (w, h), (0.0, h)],
    ]

    for corners_uv in corner_sets:
        ground_pts = []
        for (u, v) in corners_uv:
            d_w = _pixel_ray_to_world_dir(u, v, cam_cfg, r_wc)
            p = _intersect_ground(cam_pos, d_w, z_ground=z_ground)
            if p is None:
                ground_pts = []
                break
            ground_pts.append(p)
        if len(ground_pts) == 4:
            return ground_pts

    return []


def draw_camera_debug(env, follower: DotsSplinePidFollower, z_ground=0.02):  # 카메라 시각화
    if env.viewer is None:
        return
    viz = follower.viz

    for i in range(env.num_envs):
        cam_pos = follower.debug_cam_pos_w[i]
        r_wc = follower.debug_cam_r_wc[i]

        _draw_cross_thick(
            env=env,
            env_handle=env.envs[i],
            x=float(cam_pos[0]),
            y=float(cam_pos[1]),
            z=float(cam_pos[2]),
            half_len=viz.cam_pos_cross_half_len,
            color=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            thickness=viz.cam_pos_cross_thickness,
        )

        ground_pts = _project_frame_to_ground(
            cam_pos=cam_pos,
            r_wc=r_wc,
            cam_cfg=follower.cam_cfg,
            z_ground=z_ground,
        )

        if len(ground_pts) == 4:
            rect_col = np.array([1.0, 0.5, 0.0], dtype=np.float32)
            for k in range(4):
                p0 = ground_pts[k]
                p1 = ground_pts[(k + 1) % 4]
                _draw_segment_thick(
                    env,
                    env.envs[i],
                    p0,
                    p1,
                    rect_col,
                    width=viz.cam_frame_width,
                    n_lines=viz.cam_frame_n_lines,
                )

            ray_col = np.array([1.0, 0.3, 0.3], dtype=np.float32)
            for p in ground_pts:
                env.gym.add_lines(
                    env.viewer,
                    env.envs[i],
                    1,
                    np.concatenate([cam_pos.astype(np.float32), p]).astype(np.float32),
                    ray_col,
                )


def draw_command_arrows(env, follower: DotsSplinePidFollower, commands, z_offset=0.58):  # 명령 화살표 시각화
    if env.viewer is None:
        return
    if commands is None:
        return

    _roll, _pitch, yaw = quat_to_rpy(env.base_quat)
    max_vx = max(float(follower.vx_max), 1e-6)
    viz = follower.viz

    for i in range(env.num_envs):
        base = env.root_states[i, 0:3].detach().cpu().numpy().astype(np.float32)
        vx_cmd = float(commands[i, 0].item())
        wz_cmd = float(commands[i, 2].item())
        yaw_i = float(yaw[i].item())

        heading = yaw_i + wz_cmd

        speed_ratio = float(np.clip(abs(vx_cmd) / max_vx, 0.0, 1.0))
        arrow_len = 0.16 + 0.55 * speed_ratio
        head_len = max(0.08, 0.22 * arrow_len)
        wing_angle = math.radians(28.0)

        start = np.array([base[0], base[1], base[2] + z_offset], dtype=np.float32)
        end = start + np.array(
            [
                arrow_len * math.cos(heading),
                arrow_len * math.sin(heading),
                0.0,
            ],
            dtype=np.float32,
        )

        color = (
            np.array([1.0, 0.10, 0.10], dtype=np.float32)
            if bool(follower.in_recovery[i].item())
            else np.array([0.10, 1.0, 0.10], dtype=np.float32)
        )

        _draw_segment_thick(
            env,
            env.envs[i],
            start,
            end,
            color,
            width=viz.cmd_arrow_body_width,
            n_lines=viz.cmd_arrow_body_n_lines,
        )

        left_tip = end - np.array(
            [
                head_len * math.cos(heading - wing_angle),
                head_len * math.sin(heading - wing_angle),
                0.0,
            ],
            dtype=np.float32,
        )
        right_tip = end - np.array(
            [
                head_len * math.cos(heading + wing_angle),
                head_len * math.sin(heading + wing_angle),
                0.0,
            ],
            dtype=np.float32,
        )

        _draw_segment_thick(
            env,
            env.envs[i],
            end,
            left_tip,
            color,
            width=viz.cmd_arrow_head_width,
            n_lines=viz.cmd_arrow_head_n_lines,
        )
        _draw_segment_thick(
            env,
            env.envs[i],
            end,
            right_tip,
            color,
            width=viz.cmd_arrow_head_width,
            n_lines=viz.cmd_arrow_head_n_lines,
        )


def print_status(step_idx, env, commands, follower: DotsSplinePidFollower, every=50):  # 주기적으로 핵심 상태 로그 출력
    if step_idx % every != 0:
        return

    recover0 = bool(follower.in_recovery[0].item())
    nvis0 = int(follower.n_visible[0].item())
    uerr0 = float(follower.vision_u_err[0].item())
    uerr_la0 = float(follower.vision_u_err_la[0].item())
    slope0 = float(follower.vision_slope[0].item())
    mode0 = "RECOV" if recover0 else "TRACK"
    t_sec = step_idx * float(env.dt)
    use_la0 = bool(getattr(follower, "use_lookahead", False))

    alpha_normal = float(np.clip(getattr(follower, "lookahead_alpha_normal", 0.0), 0.0, 1.0))
    alpha_recovery = float(np.clip(getattr(follower, "lookahead_alpha_recovery", 0.0), 0.0, 1.0))
    alpha0 = alpha_recovery if recover0 else alpha_normal
    if (not use_la0) or (nvis0 < 2):
        alpha0 = 0.0
    uctrl0 = (1.0 - alpha0) * uerr0 + alpha0 * uerr_la0

    print(
        f"[{step_idx} steps] {t_sec:4.2f}s // {mode0:4} / {nvis0:1d} dots / a: {alpha0:4.2f} \n"
        f"   e_near: {uerr0:+6.3f} | e_la: {uerr_la0:+6.3f} | e_ctrl: {uctrl0:+6.3f}\n"
        f"   slope: {math.degrees(math.atan(120 * slope0)):+7.3f} deg\n"
        f"   CMD: ({commands[0, 0].item():+6.3f}, {commands[0, 2].item():+6.3f}) | "
        f"({env.commands[0, 0].item():+6.3f}, {env.commands[0, 2].item():+6.3f})"
    )




# 추종 민감도: k_u, k_slope
# 안정성/진동: dv_max, dw_max, hold_steps
# 속도 프로파일: v_base, k_v_u, k_v_slope, vx_max
# 시야 불량 대응: recover_*, per_pt_dropout_prob, pixel_jitter_std, max_centers
# 경로 난이도: manual_block_specs, manual_*_jitter, block_ds

# in_recovery: env별 recovery 상태 bool.
# v_cmd_prev, w_cmd_prev: 이전 step 명령(레이트 리밋 기준).
# v_cmd_hold, w_cmd_hold: hold_steps 동안 유지되는 명령.
# centers_uv: 선택된 점들의 이미지 좌표.
# centers_valid: centers 유효 마스크.
# vision_u_err: 하단 점 기준 횡오차(정규화).
# vision_slope: 점선 기울기 특징.
# n_visible: 현재 가시 점 개수.
# visible_local_points: 디버그용 로컬 좌표 점들.
# debug_cam_pos_w, debug_cam_r_wc: 카메라 월드 위치/회전(디버그 드로잉용).

# wrap_to_pi_torch(x): 각도 텐서를 [-pi, pi]로 래핑.
# quat_to_rpy(quat): quat[N,4] → (roll,pitch,yaw) 텐서.
# get_local_pose_rpy(env): env 상태에서 로컬 위치와 rpy 추출.
# perturb_initial_pose(env, y_range=0.20, yaw_range=0.30): 초기 y/yaw 랜덤 perturb.
# reset_done_envs(env, done_ids, follower): done env 재배치 + follower 내부 상태 초기화.
# arrange_envs_along_y(env, y_gap=2.6): env 2개를 y축으로 떨어뜨려 배치.
# rot_x(a), rot_y(a), rot_z(a): 축 회전행렬(입력 rad).
# rpy_to_rot(roll,pitch,yaw): ZYX 회전행렬 조합.
# _catmull_rom_chain(pts_xy, samples_per_seg): 웨이포인트 spline 밀집 샘플링.
# _compute_heading_curvature(path_xy, s): 경로 heading/curvature 계산.
# _make_waypoints(env_idx): non-block용 랜덤 웨이포인트 생성.
# _get_manual_block_specs(env_idx): env 인덱스로 manual spec 선택.
# _apply_jitter_to_block_spec(block_spec): 블록 한 개에 지터 적용.
# _append_straight_block(pts,x,y,heading,length): 흔들림 포함 직선점 추가.
# _append_turn_block(pts,x,y,heading,radius,turn_angle,direction): 원호 점 추가.
# _generate_block_path_points(env_idx): block 경로 전체 점 생성.
# _build_dash_points(path_xy,path_s): dash/gap 마스크로 점선 점 추출.
# setup_random_dotted_spline_paths(): env별 path/s/heading/curvature/dash cache 생성.
# _camera_pose_world(base_pos_w,roll,pitch,yaw): 카메라 월드 자세 계산.
# _project_world_points(world_pts,cam_pos,r_wc): 월드점→영상 투영 + 유효점 필터.
# update_perception(local_x,local_y,base_z,roll,pitch,yaw,env_origins): 비전 특징 갱신 핵심.
# _update_recovery_state_vision(): 진입/해제 히스테리시스.
# _clip_and_rate_limit(v_raw,w_raw): 한계 + 변화율 제한.
# compute_upper_command_from_vision(step_idx): 최종 commands[num_envs,4] 생성.
# reset_env_state(done_ids): follower 내부 상태 리셋.
# print_config(): 현재 설정 출력.
# _draw_cross_thick(env, env_handle, x, y, z, half_len, color, thickness=0.008): 십자 마커 그리기.
# draw_path_and_dashes(env, follower): 점선 경로 시각화.
# draw_tracking_points(env, follower): 가시점 시각화.
# _pixel_ray_to_world_dir(u,v,cam_cfg,r_wc): 픽셀 광선 방향 계산.
# _intersect_ground(cam_pos,dir_world,z_ground): 광선-지면 교차.
# _project_frame_to_ground(cam_pos,r_wc,cam_cfg,z_ground): 화면 프러스텀 사각형의 지면 투영점 계산.
# draw_camera_debug(env,follower,z_ground=0.02): 카메라 위치/FOV 디버그 시각화.
# print_status(step_idx, env, commands, follower, every=50): 상태 로그 출력.

# def __init__(self, num_envs, device, env_dt, seed=7):
#         self.num_envs = num_envs
#         self.device = device
#         self.env_dt = env_dt
#         self.rng = np.random.default_rng(seed)

#         # ----- command envelope -----
#         self.vx_min = 0.10 # 속도 범위
#         self.vx_max = 1.20
#         self.wz_min = -0.90
#         self.wz_max = 0.90

#         self.hold_steps = 3 # 3스텝마다 명령 업뎃
#         self.dv_max = 0.12 # 스텝당 최대 속도 변화량
#         self.dw_max = 0.0

#         # ----- vision-only controller -----
#         self.v_base = 0.85 # 평소 속도

#         self.k_u = 1.40   # y편차에 대한 yaw 보정 게인
#         self.k_slope = 0.90  # 기울기에 대한 yaw 보정 게인
#         self.k_v_u = 0.35  # y편차에 대한 v 보정 게인
#         self.k_v_slope = 0.25  # 기울기에 대한 v 보정 게인

#         # ----- recovery (vision only) -----
#         self.recover_enter_nvis = 1.0  # 점이 1개 이하로 보이면 복귀 모드 진입
#         self.recover_exit_nvis = 3.0 # 점이 3개 이상으로 보이면 복귀 모드 탈출
#         self.recover_enter_u = 0.70 # y편차 절대값이 0.70 이상이면 복귀 모드 진입
#         self.recover_exit_u = 0.35 # y편차 절대값이 0.35 이하이면 복귀 모드 탈출

#         self.recover_vx = 0.12 # 복귀 모드에서의 x축 속도 고정
#         self.recover_wz = 0.75 # 복귀 모드에서의 z축 회전 속도 고정

#         # ----- camera/perception -----
#         self.cam_cfg = CameraCfg()
#         self.use_rp_stabilization = False # 롤/피치 안정화 여부 (실제 카메라는 고정이지만 시점 보정을 할지)
#         self.imu_noise_std_deg = 0.0 # imu 노이즈 표준편차 (degrees)

#         self.max_centers = 8 # 한 번에 인식 가능한 점 최대 개수 (카메라 시야 내에서)
#         self.per_pt_dropout_prob = 0.12 # 각 점마다 독립적으로 사라질 확률 (카메라 인식 실패 시뮬레이션)
#         self.burst_dropout_prob = 0.04 # 한 프레임 전체가 인식 실패할 확률 (카메라 프레임 드롭 시뮬레이션)
#         self.pixel_jitter_std = 1.2 # 점 위치에 픽셀 단위로 가우시안 노이즈 추가 (카메라 측정 잡음 시뮬레이션)

#         # ----- waypoint 모드일 때 -----
#         self.num_waypoints = 26 # 웨이포인트 개수 (대략 하나당 1.3m)
#         self.samples_per_seg = 36 # 웨이포인트 사이를 보간하여 생성할 점 개수 (대략 0.035m 간격)
#         self.path_draw_stride = 6 # 시각화할 때 경로 점을 몇 개씩 건너뛸지 (값이 작을수록 더 촘촘히 그려짐)
#         self.path_z = 0.002 # 경로 시각화할 때 z 높이 (바닥에서 약간 띄워서 그리면 더 잘 보임)
#         self.dash_len = 0.28 # 흰색 점선 길이
#         self.gap_len = 0.22 # 흰색 점선 사이 간격

#         self.waypoints = None
#         self.path_points = None
#         self.path_s = None
#         self.path_heading = None
#         self.path_curvature = None
#         self.dash_points = []

#         # ----- block 모드일 때 -----
#         self.path_mode = "block"

#         self.block_ds = 0.20          # 흰색 점선 간격

#         # ----- 직선 구간 흔들림 -----
#         self.straight_wiggle_amp_min = 0.12   # meters
#         self.straight_wiggle_amp_max = 0.20   # meters
#         self.straight_wiggle_freq_min = 3.0
#         self.straight_wiggle_freq_max = 5.0
#         self.straight_wiggle_phase_random = True
