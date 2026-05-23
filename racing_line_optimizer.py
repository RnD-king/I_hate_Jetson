#!/usr/bin/env python3
"""Offline racing-line optimizer for the ROS 2 Racing Challenge.

This script does not modify the assignment package.  It builds a spatial
reference line for the fixed stadium track, then computes friction-dependent
speed profiles on that line.

The optimizer uses the same constants that appear in racing_sim.py:

- stadium centerline radius: 5.0 m
- straight half length: 10.0 m
- track half width: 2.0 m
- wheelbase: 1.0 m
- vehicle width and overhangs
- acceleration, braking, steering, and steering-rate limits
- obstacle geometry
- tire static friction and 85-100 percent surface multiplier

The problem is nonlinear and non-convex, so this is a constrained numerical
optimizer with multi-start search and a strict post-solve verifier.  It should
be treated as an offline planner: generate a candidate, inspect the plot/CSV,
then embed the resulting table into racing_controller.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


Array = np.ndarray


@dataclass(frozen=True)
class TrackParams:
    straight_half_length: float = 10.0
    centerline_radius: float = 5.0
    track_half_width: float = 2.0

    @property
    def outer_radius(self) -> float:
        return self.centerline_radius + self.track_half_width

    @property
    def inner_radius(self) -> float:
        return self.centerline_radius - self.track_half_width

    @property
    def length(self) -> float:
        return (
            4.0 * self.straight_half_length
            + 2.0 * math.pi * self.centerline_radius
        )


@dataclass(frozen=True)
class VehicleParams:
    wheelbase: float = 1.0
    front_overhang: float = 0.35
    rear_overhang: float = 0.25
    vehicle_width: float = 0.65
    max_speed: float = 9.0
    max_steering: float = 0.60
    max_accel: float = 2.8
    max_decel: float = 4.5
    max_steering_rate: float = 6.0
    gravity: float = 9.81
    base_static_friction_coefficient: float = 0.85

    @property
    def half_width(self) -> float:
        return self.vehicle_width / 2.0

    @property
    def front_x(self) -> float:
        return self.wheelbase + self.front_overhang

    @property
    def rear_x(self) -> float:
        return -self.rear_overhang

    @property
    def max_curvature_from_steering(self) -> float:
        return math.tan(self.max_steering) / self.wheelbase


@dataclass(frozen=True)
class Obstacle:
    center_x: float
    center_y: float
    size_x: float
    size_y: float


@dataclass
class PathData:
    s_center: Array
    x: Array
    y: Array
    yaw: Array
    curvature: Array
    delta: Array
    d_delta_ds: Array
    segment_length: Array
    lateral_offset: Array
    control_offsets: Array


@dataclass
class SpeedProfile:
    friction_percent: int
    effective_mu: float
    speed: Array
    acceleration: Array
    steering_rate: Array
    lap_time: float


@dataclass
class EvalResult:
    objective: float
    lap_time: float
    penalty: float
    path: PathData
    profile: SpeedProfile
    violations: Dict[str, float]


TRACK = TrackParams()
VEHICLE = VehicleParams()
OBSTACLES = (
    Obstacle(4.0, TRACK.centerline_radius + 1.00, 0.5, 2.0),
    Obstacle(-4.0, TRACK.centerline_radius - 1.00, 0.5, 2.0),
)


def wrap_angle(angle: Array) -> Array:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def cyclic_progress_delta(a: Array, b: float, track_length: float) -> Array:
    return (a - b + 0.5 * track_length) % track_length - 0.5 * track_length


def centerline_pose(progress: Array, track: TrackParams = TRACK) -> Tuple[Array, Array, Array]:
    s = np.mod(progress, track.length)
    x = np.empty_like(s, dtype=float)
    y = np.empty_like(s, dtype=float)
    yaw = np.empty_like(s, dtype=float)

    a = track.straight_half_length
    r = track.centerline_radius
    semicircle = math.pi * r
    top_straight = 2.0 * a

    mask = s <= a
    x[mask] = s[mask]
    y[mask] = -r
    yaw[mask] = 0.0

    mask = (s > a) & (s <= a + semicircle)
    arc = s[mask] - a
    angle = -math.pi / 2.0 + arc / r
    x[mask] = a + r * np.cos(angle)
    y[mask] = r * np.sin(angle)
    yaw[mask] = angle + math.pi / 2.0

    mask = (s > a + semicircle) & (s <= a + semicircle + top_straight)
    straight = s[mask] - a - semicircle
    x[mask] = a - straight
    y[mask] = r
    yaw[mask] = math.pi

    mask = s > a + semicircle + top_straight
    arc = s[mask] - a - semicircle - top_straight
    left_arc = arc <= semicircle
    idx = np.flatnonzero(mask)

    arc_idx = idx[left_arc]
    angle = math.pi / 2.0 + arc[left_arc] / r
    x[arc_idx] = -a + r * np.cos(angle)
    y[arc_idx] = r * np.sin(angle)
    yaw[arc_idx] = angle + math.pi / 2.0

    straight_idx = idx[~left_arc]
    straight = arc[~left_arc] - semicircle
    x[straight_idx] = -a + straight
    y[straight_idx] = -r
    yaw[straight_idx] = 0.0

    return x, y, wrap_angle(yaw)


def capsule_sdf(x: Array, y: Array, radius: float, track: TrackParams = TRACK) -> Array:
    qx = np.maximum(np.abs(x) - track.straight_half_length, 0.0)
    return np.hypot(qx, y) - radius


def rectangle_sdf(x: Array, y: Array, obstacle: Obstacle) -> Array:
    qx = np.abs(x - obstacle.center_x) - obstacle.size_x / 2.0
    qy = np.abs(y - obstacle.center_y) - obstacle.size_y / 2.0
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def vehicle_corners(path: PathData, vehicle: VehicleParams = VEHICLE) -> List[Tuple[Array, Array]]:
    cos_yaw = np.cos(path.yaw)
    sin_yaw = np.sin(path.yaw)
    local_points = (
        (vehicle.front_x, vehicle.half_width),
        (vehicle.front_x, -vehicle.half_width),
        (vehicle.rear_x, -vehicle.half_width),
        (vehicle.rear_x, vehicle.half_width),
    )
    corners = []
    for local_x, local_y in local_points:
        world_x = path.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = path.y + local_x * sin_yaw + local_y * cos_yaw
        corners.append((world_x, world_y))
    return corners


def initial_offset_controls(control_count: int, scale: float = 1.0) -> Array:
    s_ctrl = np.linspace(0.0, TRACK.length, control_count, endpoint=False)
    d = np.zeros(control_count, dtype=float)

    # Top straight obstacle at x=4 blocks upper lane; with yaw=pi, positive
    # offset moves downward, so pass it with d > 0.
    s_obs_upper = (
        TRACK.straight_half_length
        + math.pi * TRACK.centerline_radius
        + (TRACK.straight_half_length - 4.0)
    )
    # Top straight obstacle at x=-4 blocks lower lane; pass it with d < 0.
    s_obs_lower = (
        TRACK.straight_half_length
        + math.pi * TRACK.centerline_radius
        + (TRACK.straight_half_length + 4.0)
    )
    d += 1.00 * np.exp(-(cyclic_progress_delta(s_ctrl, s_obs_upper, TRACK.length) / 3.2) ** 2)
    d += -1.00 * np.exp(-(cyclic_progress_delta(s_ctrl, s_obs_lower, TRACK.length) / 3.2) ** 2)

    # Give the optimizer a mild outside-inside-outside bias around the two
    # semicircles without forcing the final line.
    right_mid = TRACK.straight_half_length + 0.5 * math.pi * TRACK.centerline_radius
    left_mid = (
        3.0 * TRACK.straight_half_length
        + 1.5 * math.pi * TRACK.centerline_radius
    )
    d += -0.35 * np.exp(-(cyclic_progress_delta(s_ctrl, right_mid, TRACK.length) / 6.0) ** 2)
    d += -0.35 * np.exp(-(cyclic_progress_delta(s_ctrl, left_mid, TRACK.length) / 6.0) ** 2)

    return scale * d


def build_path(control_offsets: Array, sample_count: int) -> PathData:
    control_count = len(control_offsets)
    s_ctrl = np.linspace(0.0, TRACK.length, control_count + 1)
    d_ctrl = np.concatenate([control_offsets, control_offsets[:1]])
    spline = CubicSpline(s_ctrl, d_ctrl, bc_type="periodic")

    s = np.linspace(0.0, TRACK.length, sample_count, endpoint=False)
    offset = spline(s)
    cx, cy, cyaw = centerline_pose(s)
    normal_x = -np.sin(cyaw)
    normal_y = np.cos(cyaw)
    x = cx + offset * normal_x
    y = cy + offset * normal_y

    ds_center = TRACK.length / sample_count
    x_f = np.roll(x, -1)
    x_b = np.roll(x, 1)
    y_f = np.roll(y, -1)
    y_b = np.roll(y, 1)
    dx = (x_f - x_b) / (2.0 * ds_center)
    dy = (y_f - y_b) / (2.0 * ds_center)
    ddx = (x_f - 2.0 * x + x_b) / (ds_center * ds_center)
    ddy = (y_f - 2.0 * y + y_b) / (ds_center * ds_center)

    yaw = np.unwrap(np.arctan2(dy, dx))
    curvature = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    delta = np.arctan(VEHICLE.wheelbase * curvature)

    segment_length = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    prev_segment = np.roll(segment_length, 1)
    d_delta_ds = (np.roll(delta, -1) - np.roll(delta, 1)) / np.maximum(
        segment_length + prev_segment,
        1e-6,
    )

    return PathData(
        s_center=s,
        x=x,
        y=y,
        yaw=wrap_angle(yaw),
        curvature=curvature,
        delta=delta,
        d_delta_ds=d_delta_ds,
        segment_length=segment_length,
        lateral_offset=offset,
        control_offsets=control_offsets.copy(),
    )


def speed_profile(
    path: PathData,
    friction_percent: int,
    friction_reserve: float,
    iterations: int = 16,
) -> SpeedProfile:
    surface_multiplier = friction_percent / 100.0
    effective_mu = VEHICLE.base_static_friction_coefficient * surface_multiplier
    friction_accel = effective_mu * VEHICLE.gravity * friction_reserve
    abs_kappa = np.abs(path.curvature)

    v_limit = np.full_like(abs_kappa, VEHICLE.max_speed, dtype=float)
    curved = abs_kappa > 1e-6
    v_limit[curved] = np.minimum(
        v_limit[curved],
        np.sqrt(friction_accel / np.maximum(abs_kappa[curved], 1e-9)),
    )

    steer_rate_sensitive = np.abs(path.d_delta_ds) > 1e-5
    v_limit[steer_rate_sensitive] = np.minimum(
        v_limit[steer_rate_sensitive],
        VEHICLE.max_steering_rate / np.abs(path.d_delta_ds[steer_rate_sensitive]),
    )
    v_limit = np.maximum(v_limit, 0.4)

    v = v_limit.copy()
    n = len(v)

    for _ in range(iterations):
        for i in range(2 * n):
            j = i % n
            k = (j + 1) % n
            ds = max(path.segment_length[j], 1e-6)
            lat_accel = abs(path.curvature[j]) * v[j] * v[j]
            long_limit = math.sqrt(max(friction_accel * friction_accel - lat_accel * lat_accel, 0.0))
            accel = min(VEHICLE.max_accel, long_limit)
            reachable = math.sqrt(max(v[j] * v[j] + 2.0 * accel * ds, 0.0))
            if v[k] > reachable:
                v[k] = reachable

        for i in range(2 * n - 1, -1, -1):
            j = i % n
            k = (j + 1) % n
            ds = max(path.segment_length[j], 1e-6)
            lat_accel = abs(path.curvature[k]) * v[k] * v[k]
            long_limit = math.sqrt(max(friction_accel * friction_accel - lat_accel * lat_accel, 0.0))
            decel = min(VEHICLE.max_decel, long_limit)
            reachable = math.sqrt(max(v[k] * v[k] + 2.0 * decel * ds, 0.0))
            if v[j] > reachable:
                v[j] = reachable

        v = np.minimum(v, v_limit)

    v_next = np.roll(v, -1)
    ds = np.maximum(path.segment_length, 1e-6)
    acceleration = (v_next * v_next - v * v) / (2.0 * ds)
    steering_rate = path.d_delta_ds * v
    avg_speed = np.maximum(0.5 * (v + v_next), 0.1)
    lap_time = float(np.sum(ds / avg_speed))
    return SpeedProfile(
        friction_percent=friction_percent,
        effective_mu=effective_mu,
        speed=v,
        acceleration=acceleration,
        steering_rate=steering_rate,
        lap_time=lap_time,
    )


def violation_metrics(
    path: PathData,
    profile: SpeedProfile,
    track_margin: float,
    obstacle_margin: float,
    friction_reserve: float,
) -> Dict[str, float]:
    corners = vehicle_corners(path)
    outer_violation = 0.0
    inner_violation = 0.0
    obstacle_violation = 0.0

    for cx, cy in corners:
        outer_sdf = capsule_sdf(cx, cy, TRACK.outer_radius)
        inner_sdf = capsule_sdf(cx, cy, TRACK.inner_radius)
        outer_violation = max(outer_violation, float(np.max(outer_sdf + track_margin)))
        inner_violation = max(inner_violation, float(np.max(track_margin - inner_sdf)))

        for obstacle in OBSTACLES:
            obs_sdf = rectangle_sdf(cx, cy, obstacle)
            obstacle_violation = max(
                obstacle_violation,
                float(np.max(obstacle_margin - obs_sdf)),
            )

    steering_violation = float(np.max(np.abs(path.delta) - VEHICLE.max_steering))
    steering_rate_violation = float(
        np.max(np.abs(profile.steering_rate) - VEHICLE.max_steering_rate)
    )
    accel_violation = float(np.max(profile.acceleration - VEHICLE.max_accel))
    decel_violation = float(np.max(-profile.acceleration - VEHICLE.max_decel))
    speed_violation = float(np.max(profile.speed - VEHICLE.max_speed))

    friction_accel = profile.effective_mu * VEHICLE.gravity * friction_reserve
    lat_accel = profile.speed * profile.speed * np.abs(path.curvature)
    combined_accel = np.hypot(lat_accel, np.maximum(profile.acceleration, 0.0))
    friction_violation = float(np.max(combined_accel - friction_accel))

    return {
        "outer_track_m": max(0.0, outer_violation),
        "inner_track_m": max(0.0, inner_violation),
        "obstacle_m": max(0.0, obstacle_violation),
        "steering_rad": max(0.0, steering_violation),
        "steering_rate_rad_s": max(0.0, steering_rate_violation),
        "accel_m_s2": max(0.0, accel_violation),
        "decel_m_s2": max(0.0, decel_violation),
        "speed_m_s": max(0.0, speed_violation),
        "friction_m_s2": max(0.0, friction_violation),
    }


def corner_arc_regularization_cost(
    path: PathData,
    arc_const_weight: float,
    transition_smooth_weight: float,
    transition_length: float,
    entry_extension: float,
    exit_extension: float,
) -> float:
    if arc_const_weight <= 0.0 and transition_smooth_weight <= 0.0:
        return 0.0

    right_start = TRACK.straight_half_length
    right_end = right_start + math.pi * TRACK.centerline_radius
    left_start = right_end + 2.0 * TRACK.straight_half_length
    left_end = left_start + math.pi * TRACK.centerline_radius

    s = path.s_center
    curvature = path.curvature
    dkappa_ds = (
        np.roll(curvature, -1) - curvature
    ) / np.maximum(path.segment_length, 1e-6)
    cost = 0.0

    for base_start, base_end in (
        (right_start, right_end),
        (left_start, left_end),
    ):
        start = max(0.0, base_start - max(entry_extension, 0.0))
        end = min(
            TRACK.length,
            base_end + max(exit_extension, 0.0),
        )
        corner_length = end - start
        blend = min(max(transition_length, 0.0), 0.35 * corner_length)
        core_mask = (s >= start + blend) & (s <= end - blend)
        transition_mask = (
            ((s >= start - blend) & (s <= start + blend)) |
            ((s >= end - blend) & (s <= end + blend))
        )

        if arc_const_weight > 0.0 and np.count_nonzero(core_mask) >= 3:
            core_curvature = curvature[core_mask]
            mean_curvature = float(np.mean(core_curvature))
            cost += arc_const_weight * float(
                np.mean((core_curvature - mean_curvature) ** 2)
            )

        if (
            transition_smooth_weight > 0.0 and
            np.count_nonzero(transition_mask) >= 3
        ):
            transition_slope = dkappa_ds[transition_mask]
            cost += transition_smooth_weight * float(
                np.mean(transition_slope * transition_slope)
            )

    return cost


def left_exit_out_regularization_cost(
    path: PathData,
    weight: float,
    target_offset: float,
    start_progress: float,
    full_progress: float,
    end_progress: float,
) -> float:
    if weight <= 0.0:
        return 0.0
    if not (start_progress < full_progress <= end_progress):
        return 0.0

    s = path.s_center
    mask = (s >= start_progress) & (s <= end_progress)
    if np.count_nonzero(mask) < 3:
        return 0.0

    local_s = s[mask]
    ramp_ratio = np.clip(
        (local_s - start_progress) / max(full_progress - start_progress, 1e-6),
        0.0,
        1.0,
    )
    ramp = ramp_ratio * ramp_ratio * (3.0 - 2.0 * ramp_ratio)
    offset_error = np.maximum(
        path.lateral_offset[mask] - target_offset,
        0.0,
    )
    return weight * float(np.mean((ramp * offset_error) ** 2))


def left_exit_curvature_continuation_cost(
    path: PathData,
    weight: float,
    source_start_progress: float,
    source_end_progress: float,
    target_start_progress: float,
    target_end_progress: float,
    end_scale: float,
) -> float:
    if weight <= 0.0:
        return 0.0
    if not (
        source_start_progress < source_end_progress <= target_start_progress <
        target_end_progress
    ):
        return 0.0

    s = path.s_center
    source_mask = (
        (s >= source_start_progress) &
        (s <= source_end_progress)
    )
    target_mask = (
        (s >= target_start_progress) &
        (s <= target_end_progress)
    )
    if np.count_nonzero(source_mask) < 3 or np.count_nonzero(target_mask) < 3:
        return 0.0

    source_curvature = float(np.mean(path.curvature[source_mask]))
    local_s = s[target_mask]
    ratio = np.clip(
        (
            local_s - target_start_progress
        ) / max(target_end_progress - target_start_progress, 1e-6),
        0.0,
        1.0,
    )
    taper = ratio * ratio * (3.0 - 2.0 * ratio)
    target_curvature = source_curvature * (
        (1.0 - taper) + end_scale * taper
    )
    curvature_error = path.curvature[target_mask] - target_curvature
    return weight * float(np.mean(curvature_error * curvature_error))


def evaluate(
    control_offsets: Array,
    sample_count: int,
    friction_percent: int,
    track_margin: float,
    obstacle_margin: float,
    friction_reserve: float,
    control_smooth_weight: float,
    curvature_smooth_weight: float,
    steering_rate_soft_weight: float,
    steering_rate_soft_limit: float,
    offset_bound_weight: float,
    corner_arc_const_weight: float,
    corner_transition_smooth_weight: float,
    corner_transition_length: float,
    corner_entry_extension: float,
    corner_exit_extension: float,
    left_exit_out_weight: float,
    left_exit_target_offset: float,
    left_exit_start_progress: float,
    left_exit_full_progress: float,
    left_exit_end_progress: float,
    left_exit_curvature_weight: float,
    left_exit_curvature_source_start: float,
    left_exit_curvature_source_end: float,
    left_exit_curvature_target_start: float,
    left_exit_curvature_target_end: float,
    left_exit_curvature_end_scale: float,
) -> EvalResult:
    path = build_path(control_offsets, sample_count)
    profile = speed_profile(path, friction_percent, friction_reserve)
    violations = violation_metrics(
        path,
        profile,
        track_margin=track_margin,
        obstacle_margin=obstacle_margin,
        friction_reserve=friction_reserve,
    )

    penalty = 0.0
    penalty += 8.0e5 * violations["outer_track_m"] ** 2
    penalty += 8.0e5 * violations["inner_track_m"] ** 2
    penalty += 8.0e5 * violations["obstacle_m"] ** 2
    penalty += 2.0e4 * violations["steering_rad"] ** 2
    penalty += 5.0e3 * violations["steering_rate_rad_s"] ** 2
    penalty += 5.0e3 * violations["accel_m_s2"] ** 2
    penalty += 5.0e3 * violations["decel_m_s2"] ** 2
    penalty += 5.0e3 * violations["friction_m_s2"] ** 2

    offset_overshoot = np.maximum(
        np.abs(path.lateral_offset) - np.max(np.abs(control_offsets)),
        0.0,
    )
    penalty += offset_bound_weight * float(np.sum(offset_overshoot * offset_overshoot))

    steering_rate_soft_excess = np.maximum(
        np.abs(profile.steering_rate) - steering_rate_soft_limit,
        0.0,
    )
    penalty += steering_rate_soft_weight * float(
        np.mean(steering_rate_soft_excess * steering_rate_soft_excess)
    )

    # Soft regularization suppresses local wiggles that are legal but poor for
    # the simulator's slip-based tire model.
    d2 = np.roll(control_offsets, -1) - 2.0 * control_offsets + np.roll(control_offsets, 1)
    dkappa = np.roll(path.curvature, -1) - path.curvature
    smooth_cost = (
        control_smooth_weight * float(np.sum(d2 * d2)) +
        curvature_smooth_weight * float(np.sum(dkappa * dkappa))
    )
    smooth_cost += corner_arc_regularization_cost(
        path,
        corner_arc_const_weight,
        corner_transition_smooth_weight,
        corner_transition_length,
        corner_entry_extension,
        corner_exit_extension,
    )
    smooth_cost += left_exit_out_regularization_cost(
        path,
        left_exit_out_weight,
        left_exit_target_offset,
        left_exit_start_progress,
        left_exit_full_progress,
        left_exit_end_progress,
    )
    smooth_cost += left_exit_curvature_continuation_cost(
        path,
        left_exit_curvature_weight,
        left_exit_curvature_source_start,
        left_exit_curvature_source_end,
        left_exit_curvature_target_start,
        left_exit_curvature_target_end,
        left_exit_curvature_end_scale,
    )

    objective = profile.lap_time + penalty + smooth_cost
    return EvalResult(
        objective=float(objective),
        lap_time=profile.lap_time,
        penalty=float(penalty),
        path=path,
        profile=profile,
        violations=violations,
    )


def optimize_line(args: argparse.Namespace) -> EvalResult:
    rng = np.random.default_rng(args.seed)
    bounds = [(-args.max_offset, args.max_offset)] * args.control_count

    starts: List[Array] = [
        np.zeros(args.control_count),
        initial_offset_controls(args.control_count, 0.75),
        initial_offset_controls(args.control_count, 1.00),
        initial_offset_controls(args.control_count, 1.20),
    ]
    while len(starts) < args.starts:
        base = initial_offset_controls(args.control_count, rng.uniform(0.7, 1.2))
        noise = rng.normal(0.0, args.random_offset_sigma, args.control_count)
        starts.append(np.clip(base + noise, -args.max_offset, args.max_offset))

    best: EvalResult | None = None
    best_x: Array | None = None

    def objective(x: Array) -> float:
        corner_arc_const_weight = getattr(
            args,
            "corner_arc_const_weight",
            0.0,
        )
        corner_transition_smooth_weight = getattr(
            args,
            "corner_transition_smooth_weight",
            0.0,
        )
        corner_transition_length = getattr(
            args,
            "corner_transition_length",
            2.2,
        )
        corner_entry_extension = getattr(
            args,
            "corner_entry_extension",
            0.0,
        )
        corner_exit_extension = getattr(
            args,
            "corner_exit_extension",
            0.0,
        )
        left_exit_out_weight = getattr(args, "left_exit_out_weight", 0.0)
        left_exit_target_offset = getattr(
            args,
            "left_exit_target_offset",
            -0.85,
        )
        left_exit_start_progress = getattr(
            args,
            "left_exit_start_progress",
            57.8,
        )
        left_exit_full_progress = getattr(
            args,
            "left_exit_full_progress",
            60.7,
        )
        left_exit_end_progress = getattr(
            args,
            "left_exit_end_progress",
            63.6,
        )
        left_exit_curvature_weight = getattr(
            args,
            "left_exit_curvature_weight",
            0.0,
        )
        left_exit_curvature_source_start = getattr(
            args,
            "left_exit_curvature_source_start",
            58.4,
        )
        left_exit_curvature_source_end = getattr(
            args,
            "left_exit_curvature_source_end",
            61.2,
        )
        left_exit_curvature_target_start = getattr(
            args,
            "left_exit_curvature_target_start",
            61.2,
        )
        left_exit_curvature_target_end = getattr(
            args,
            "left_exit_curvature_target_end",
            64.4,
        )
        left_exit_curvature_end_scale = getattr(
            args,
            "left_exit_curvature_end_scale",
            0.0,
        )
        return evaluate(
            x,
            sample_count=args.samples,
            friction_percent=args.plan_friction,
            track_margin=args.track_margin,
            obstacle_margin=args.obstacle_margin,
            friction_reserve=args.friction_reserve,
            control_smooth_weight=args.control_smooth_weight,
            curvature_smooth_weight=args.curvature_smooth_weight,
            steering_rate_soft_weight=args.steering_rate_soft_weight,
            steering_rate_soft_limit=args.steering_rate_soft_limit,
            offset_bound_weight=args.offset_bound_weight,
            corner_arc_const_weight=corner_arc_const_weight,
            corner_transition_smooth_weight=corner_transition_smooth_weight,
            corner_transition_length=corner_transition_length,
            corner_entry_extension=corner_entry_extension,
            corner_exit_extension=corner_exit_extension,
            left_exit_out_weight=left_exit_out_weight,
            left_exit_target_offset=left_exit_target_offset,
            left_exit_start_progress=left_exit_start_progress,
            left_exit_full_progress=left_exit_full_progress,
            left_exit_end_progress=left_exit_end_progress,
            left_exit_curvature_weight=left_exit_curvature_weight,
            left_exit_curvature_source_start=left_exit_curvature_source_start,
            left_exit_curvature_source_end=left_exit_curvature_source_end,
            left_exit_curvature_target_start=left_exit_curvature_target_start,
            left_exit_curvature_target_end=left_exit_curvature_target_end,
            left_exit_curvature_end_scale=left_exit_curvature_end_scale,
        ).objective

    for index, start in enumerate(starts[: args.starts], start=1):
        print(f"[start {index}/{args.starts}] optimizing...")
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": args.maxiter,
                "ftol": args.ftol,
                "disp": False,
            },
        )
        corner_arc_const_weight = getattr(
            args,
            "corner_arc_const_weight",
            0.0,
        )
        corner_transition_smooth_weight = getattr(
            args,
            "corner_transition_smooth_weight",
            0.0,
        )
        corner_transition_length = getattr(
            args,
            "corner_transition_length",
            2.2,
        )
        corner_entry_extension = getattr(
            args,
            "corner_entry_extension",
            0.0,
        )
        corner_exit_extension = getattr(
            args,
            "corner_exit_extension",
            0.0,
        )
        left_exit_out_weight = getattr(args, "left_exit_out_weight", 0.0)
        left_exit_target_offset = getattr(
            args,
            "left_exit_target_offset",
            -0.85,
        )
        left_exit_start_progress = getattr(
            args,
            "left_exit_start_progress",
            57.8,
        )
        left_exit_full_progress = getattr(
            args,
            "left_exit_full_progress",
            60.7,
        )
        left_exit_end_progress = getattr(
            args,
            "left_exit_end_progress",
            63.6,
        )
        left_exit_curvature_weight = getattr(
            args,
            "left_exit_curvature_weight",
            0.0,
        )
        left_exit_curvature_source_start = getattr(
            args,
            "left_exit_curvature_source_start",
            58.4,
        )
        left_exit_curvature_source_end = getattr(
            args,
            "left_exit_curvature_source_end",
            61.2,
        )
        left_exit_curvature_target_start = getattr(
            args,
            "left_exit_curvature_target_start",
            61.2,
        )
        left_exit_curvature_target_end = getattr(
            args,
            "left_exit_curvature_target_end",
            64.4,
        )
        left_exit_curvature_end_scale = getattr(
            args,
            "left_exit_curvature_end_scale",
            0.0,
        )
        candidate = evaluate(
            result.x,
            sample_count=args.samples,
            friction_percent=args.plan_friction,
            track_margin=args.track_margin,
            obstacle_margin=args.obstacle_margin,
            friction_reserve=args.friction_reserve,
            control_smooth_weight=args.control_smooth_weight,
            curvature_smooth_weight=args.curvature_smooth_weight,
            steering_rate_soft_weight=args.steering_rate_soft_weight,
            steering_rate_soft_limit=args.steering_rate_soft_limit,
            offset_bound_weight=args.offset_bound_weight,
            corner_arc_const_weight=corner_arc_const_weight,
            corner_transition_smooth_weight=corner_transition_smooth_weight,
            corner_transition_length=corner_transition_length,
            corner_entry_extension=corner_entry_extension,
            corner_exit_extension=corner_exit_extension,
            left_exit_out_weight=left_exit_out_weight,
            left_exit_target_offset=left_exit_target_offset,
            left_exit_start_progress=left_exit_start_progress,
            left_exit_full_progress=left_exit_full_progress,
            left_exit_end_progress=left_exit_end_progress,
            left_exit_curvature_weight=left_exit_curvature_weight,
            left_exit_curvature_source_start=left_exit_curvature_source_start,
            left_exit_curvature_source_end=left_exit_curvature_source_end,
            left_exit_curvature_target_start=left_exit_curvature_target_start,
            left_exit_curvature_target_end=left_exit_curvature_target_end,
            left_exit_curvature_end_scale=left_exit_curvature_end_scale,
        )
        print(
            "  success=%s objective=%.3f lap=%.3f penalty=%.3f max_violation=%.5f"
            % (
                result.success,
                candidate.objective,
                candidate.lap_time,
                candidate.penalty,
                max(candidate.violations.values()),
            )
        )
        if best is None or candidate.objective < best.objective:
            best = candidate
            best_x = result.x.copy()

    assert best is not None and best_x is not None
    # Re-evaluate at export resolution if requested.
    if args.export_samples != args.samples:
        corner_arc_const_weight = getattr(
            args,
            "corner_arc_const_weight",
            0.0,
        )
        corner_transition_smooth_weight = getattr(
            args,
            "corner_transition_smooth_weight",
            0.0,
        )
        corner_transition_length = getattr(
            args,
            "corner_transition_length",
            2.2,
        )
        corner_entry_extension = getattr(
            args,
            "corner_entry_extension",
            0.0,
        )
        corner_exit_extension = getattr(
            args,
            "corner_exit_extension",
            0.0,
        )
        left_exit_out_weight = getattr(args, "left_exit_out_weight", 0.0)
        left_exit_target_offset = getattr(
            args,
            "left_exit_target_offset",
            -0.85,
        )
        left_exit_start_progress = getattr(
            args,
            "left_exit_start_progress",
            57.8,
        )
        left_exit_full_progress = getattr(
            args,
            "left_exit_full_progress",
            60.7,
        )
        left_exit_end_progress = getattr(
            args,
            "left_exit_end_progress",
            63.6,
        )
        left_exit_curvature_weight = getattr(
            args,
            "left_exit_curvature_weight",
            0.0,
        )
        left_exit_curvature_source_start = getattr(
            args,
            "left_exit_curvature_source_start",
            58.4,
        )
        left_exit_curvature_source_end = getattr(
            args,
            "left_exit_curvature_source_end",
            61.2,
        )
        left_exit_curvature_target_start = getattr(
            args,
            "left_exit_curvature_target_start",
            61.2,
        )
        left_exit_curvature_target_end = getattr(
            args,
            "left_exit_curvature_target_end",
            64.4,
        )
        left_exit_curvature_end_scale = getattr(
            args,
            "left_exit_curvature_end_scale",
            0.0,
        )
        best = evaluate(
            best_x,
            sample_count=args.export_samples,
            friction_percent=args.plan_friction,
            track_margin=args.track_margin,
            obstacle_margin=args.obstacle_margin,
            friction_reserve=args.friction_reserve,
            control_smooth_weight=args.control_smooth_weight,
            curvature_smooth_weight=args.curvature_smooth_weight,
            steering_rate_soft_weight=args.steering_rate_soft_weight,
            steering_rate_soft_limit=args.steering_rate_soft_limit,
            offset_bound_weight=args.offset_bound_weight,
            corner_arc_const_weight=corner_arc_const_weight,
            corner_transition_smooth_weight=corner_transition_smooth_weight,
            corner_transition_length=corner_transition_length,
            corner_entry_extension=corner_entry_extension,
            corner_exit_extension=corner_exit_extension,
            left_exit_out_weight=left_exit_out_weight,
            left_exit_target_offset=left_exit_target_offset,
            left_exit_start_progress=left_exit_start_progress,
            left_exit_full_progress=left_exit_full_progress,
            left_exit_end_progress=left_exit_end_progress,
            left_exit_curvature_weight=left_exit_curvature_weight,
            left_exit_curvature_source_start=left_exit_curvature_source_start,
            left_exit_curvature_source_end=left_exit_curvature_source_end,
            left_exit_curvature_target_start=left_exit_curvature_target_start,
            left_exit_curvature_target_end=left_exit_curvature_target_end,
            left_exit_curvature_end_scale=left_exit_curvature_end_scale,
        )
    return best


def profiles_for_all_friction(
    path: PathData,
    friction_values: Iterable[int],
    friction_reserve: float,
) -> Dict[int, SpeedProfile]:
    return {
        percent: speed_profile(path, percent, friction_reserve)
        for percent in friction_values
    }


def save_outputs(
    result: EvalResult,
    all_profiles: Dict[int, SpeedProfile],
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = result.path
    main_profile = result.profile

    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as handle:
        columns = [
            "s_center",
            "x",
            "y",
            "yaw",
            "curvature",
            "steering",
            "d_delta_ds",
            "segment_length",
            "lateral_offset",
            f"speed_{main_profile.friction_percent}",
            f"accel_{main_profile.friction_percent}",
            f"steering_rate_{main_profile.friction_percent}",
        ]
        handle.write(",".join(columns) + "\n")
        for i in range(len(path.x)):
            row = [
                path.s_center[i],
                path.x[i],
                path.y[i],
                path.yaw[i],
                path.curvature[i],
                path.delta[i],
                path.d_delta_ds[i],
                path.segment_length[i],
                path.lateral_offset[i],
                main_profile.speed[i],
                main_profile.acceleration[i],
                main_profile.steering_rate[i],
            ]
            handle.write(",".join(f"{value:.9f}" for value in row) + "\n")

    npz_path = output_prefix.with_suffix(".npz")
    payload = {
        "s_center": path.s_center,
        "x": path.x,
        "y": path.y,
        "yaw": path.yaw,
        "curvature": path.curvature,
        "steering": path.delta,
        "d_delta_ds": path.d_delta_ds,
        "segment_length": path.segment_length,
        "lateral_offset": path.lateral_offset,
        "control_offsets": path.control_offsets,
    }
    for percent, profile in all_profiles.items():
        payload[f"speed_{percent}"] = profile.speed
        payload[f"acceleration_{percent}"] = profile.acceleration
        payload[f"steering_rate_{percent}"] = profile.steering_rate
        payload[f"lap_time_{percent}"] = np.array([profile.lap_time])
        payload[f"effective_mu_{percent}"] = np.array([profile.effective_mu])
    np.savez(npz_path, **payload)

    print(f"saved: {csv_path}")
    print(f"saved: {npz_path}")


def save_plot(
    result: EvalResult,
    all_profiles: Dict[int, SpeedProfile],
    output_prefix: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f"plot skipped: {exc}")
        return

    path = result.path
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]

    def stadium_points(radius: float) -> Tuple[Array, Array]:
        theta_right = np.linspace(-math.pi / 2.0, math.pi / 2.0, 120)
        theta_left = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, 120)
        xs = np.concatenate([
            np.linspace(-TRACK.straight_half_length, TRACK.straight_half_length, 120),
            TRACK.straight_half_length + radius * np.cos(theta_right),
            np.linspace(TRACK.straight_half_length, -TRACK.straight_half_length, 120),
            -TRACK.straight_half_length + radius * np.cos(theta_left),
        ])
        ys = np.concatenate([
            -radius * np.ones(120),
            radius * np.sin(theta_right),
            radius * np.ones(120),
            radius * np.sin(theta_left),
        ])
        return xs, ys

    for radius, style in ((TRACK.outer_radius, "k-"), (TRACK.inner_radius, "k-")):
        xs, ys = stadium_points(radius)
        ax.plot(xs, ys, style, linewidth=1.2)

    cx, cy, _ = centerline_pose(np.linspace(0.0, TRACK.length, 600, endpoint=False))
    ax.plot(cx, cy, color="0.65", linewidth=0.8, linestyle="--")
    ax.plot(path.x, path.y, color="tab:blue", linewidth=2.0, label="optimized line")
    for obstacle in OBSTACLES:
        ax.add_patch(
            Rectangle(
                (obstacle.center_x - obstacle.size_x / 2.0, obstacle.center_y - obstacle.size_y / 2.0),
                obstacle.size_x,
                obstacle.size_y,
                color="tab:red",
                alpha=0.75,
            )
        )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35)
    ax.set_title("Racing line")
    ax.legend(loc="best")

    ax = axes[1]
    for percent in sorted(all_profiles):
        profile = all_profiles[percent]
        ax.plot(path.s_center, profile.speed, label=f"{percent}% ({profile.lap_time:.2f}s)")
    ax.set_xlabel("centerline progress s [m]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, linewidth=0.35)
    ax.set_title("Friction-dependent speed profiles")
    ax.legend(loc="best")
    figure.tight_layout()

    plot_path = output_prefix.with_suffix(".png")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"saved: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=420)
    parser.add_argument("--export-samples", type=int, default=720)
    parser.add_argument("--control-count", type=int, default=32)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--ftol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plan-friction", type=int, default=85)
    parser.add_argument("--friction-reserve", type=float, default=0.96)
    parser.add_argument("--track-margin", type=float, default=0.03)
    parser.add_argument("--obstacle-margin", type=float, default=0.08)
    parser.add_argument("--max-offset", type=float, default=1.55)
    parser.add_argument("--random-offset-sigma", type=float, default=0.22)
    parser.add_argument("--control-smooth-weight", type=float, default=0.08)
    parser.add_argument("--curvature-smooth-weight", type=float, default=0.02)
    parser.add_argument("--steering-rate-soft-weight", type=float, default=0.0)
    parser.add_argument("--steering-rate-soft-limit", type=float, default=5.0)
    parser.add_argument("--offset-bound-weight", type=float, default=0.0)
    parser.add_argument(
        "--corner-arc-const-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight that makes right/left corner cores keep nearly "
            "constant signed curvature."
        ),
    )
    parser.add_argument(
        "--corner-transition-smooth-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight for curvature-slope spikes near corner entry "
            "and exit transition zones."
        ),
    )
    parser.add_argument(
        "--corner-transition-length",
        type=float,
        default=2.2,
        help="Length [m] excluded from the constant-curvature corner core.",
    )
    parser.add_argument(
        "--corner-entry-extension",
        type=float,
        default=0.0,
        help="Extend each corner arc regularization interval before entry [m].",
    )
    parser.add_argument(
        "--corner-exit-extension",
        type=float,
        default=0.0,
        help="Extend each corner arc regularization interval after exit [m].",
    )
    parser.add_argument("--left-exit-out-weight", type=float, default=0.0)
    parser.add_argument("--left-exit-target-offset", type=float, default=-0.85)
    parser.add_argument("--left-exit-start-progress", type=float, default=57.8)
    parser.add_argument("--left-exit-full-progress", type=float, default=60.7)
    parser.add_argument("--left-exit-end-progress", type=float, default=63.6)
    parser.add_argument("--left-exit-curvature-weight", type=float, default=0.0)
    parser.add_argument(
        "--left-exit-curvature-source-start",
        type=float,
        default=58.4,
    )
    parser.add_argument(
        "--left-exit-curvature-source-end",
        type=float,
        default=61.2,
    )
    parser.add_argument(
        "--left-exit-curvature-target-start",
        type=float,
        default=61.2,
    )
    parser.add_argument(
        "--left-exit-curvature-target-end",
        type=float,
        default=64.4,
    )
    parser.add_argument("--left-exit-curvature-end-scale", type=float, default=0.0)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Run repeated searches with consecutive random seeds.",
    )
    parser.add_argument("--output", type=Path, default=Path("racing_line_optimized"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller search for smoke tests.",
    )
    args = parser.parse_args()
    if args.quick:
        args.samples = min(args.samples, 240)
        args.export_samples = min(args.export_samples, 360)
        args.control_count = min(args.control_count, 22)
        args.starts = min(args.starts, 3)
        args.maxiter = min(args.maxiter, 35)
    return args


def main() -> None:
    args = parse_args()
    result = None
    original_seed = args.seed
    for seed_offset in range(args.seed_count):
        args.seed = original_seed + seed_offset
        if args.seed_count > 1:
            print(
                f"\n=== seed {args.seed} "
                f"({seed_offset + 1}/{args.seed_count}) ==="
            )
        candidate = optimize_line(args)
        if result is None or candidate.objective < result.objective:
            result = candidate
    assert result is not None
    args.seed = original_seed

    friction_values = range(85, 101)
    all_profiles = profiles_for_all_friction(
        result.path,
        friction_values,
        friction_reserve=args.friction_reserve,
    )

    print("\nBest 85% planning result")
    print(f"  lap_time: {result.lap_time:.3f} s")
    print(f"  objective: {result.objective:.3f}")
    print(f"  penalty: {result.penalty:.6f}")
    print("  violations:")
    for name, value in result.violations.items():
        print(f"    {name}: {value:.6f}")
    print("  friction speed profiles:")
    for percent in sorted(all_profiles):
        profile = all_profiles[percent]
        print(
            "    %3d%%: lap=%.3f s, vmax=%.3f m/s, mu_eff=%.4f"
            % (percent, profile.lap_time, float(np.max(profile.speed)), profile.effective_mu)
        )

    save_outputs(result, all_profiles, args.output)
    if not args.no_plot:
        save_plot(result, all_profiles, args.output)


if __name__ == "__main__":
    main()
