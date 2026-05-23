import argparse
import csv
from datetime import datetime
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rclpy

from ros2_racing_challenge.racing_controller import RacingController
from ros2_racing_challenge.racing_sim import RacingSimulator
from ros2_racing_challenge_msgs.msg import CarState


def make_state_msg(state) -> CarState:
    msg = CarState()
    msg.x = float(state.x)
    msg.y = float(state.y)
    msg.yaw = float(state.yaw)
    msg.speed = float(state.speed)
    msg.steering = float(state.steering)
    return msg


def tire_usage(sim: RacingSimulator, steering: float, accel: float):
    state = sim.state
    safe_speed = max(state.speed, 0.25)
    front_slip = steering - math.atan2(
        state.lateral_speed + sim.cg_to_front_axle * state.yaw_rate,
        safe_speed,
    )
    rear_slip = -math.atan2(
        state.lateral_speed - sim.cg_to_rear_axle * state.yaw_rate,
        safe_speed,
    )
    front_load = (
        sim.vehicle_mass * sim.gravity * sim.cg_to_rear_axle / sim.wheelbase
    )
    rear_load = (
        sim.vehicle_mass * sim.gravity * sim.cg_to_front_axle / sim.wheelbase
    )
    total_load = front_load + rear_load
    longitudinal_force = sim.vehicle_mass * accel
    front_fx = longitudinal_force * front_load / total_load
    rear_fx = longitudinal_force * rear_load / total_load
    front_fy = sim.front_cornering_stiffness * front_slip
    rear_fy = sim.rear_cornering_stiffness * rear_slip
    front_limit = sim.front_static_friction_coefficient * front_load
    rear_limit = sim.rear_static_friction_coefficient * rear_load
    front_usage = math.hypot(front_fx, front_fy) / max(front_limit, 1e-6)
    rear_usage = math.hypot(rear_fx, rear_fy) / max(rear_limit, 1e-6)
    return front_usage, rear_usage


def split_progress_segments(samples):
    segments = []
    current_segment = []
    previous_progress = None
    for progress, value in samples:
        if previous_progress is not None and progress < previous_progress - 1.0:
            if len(current_segment) >= 2:
                segments.append(current_segment)
            current_segment = []
        current_segment.append((progress, value))
        previous_progress = progress
    if len(current_segment) >= 2:
        segments.append(current_segment)
    return segments


def plot_speed_profiles(controller, records, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    max_lap = max(record['lap'] for record in records)
    color_map = plt.get_cmap('tab10')
    for lap in range(1, max_lap + 1):
        lap_samples = [
            (record['progress'], record['speed'])
            for record in records
            if record['lap'] == lap
        ]
        color = color_map((lap - 1) % 10)
        for segment_index, segment in enumerate(split_progress_segments(lap_samples)):
            x_values = [progress for progress, _ in segment]
            y_values = [speed for _, speed in segment]
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.45,
                label=f'Lap {lap}' if segment_index == 0 else None,
            )

    straight = controller.straight_half_length
    right_start = straight
    right_end = straight + math.pi * controller.centerline_radius
    top_end = right_end + 2.0 * straight
    left_end = top_end + math.pi * controller.centerline_radius
    for x_position, label in (
        (right_start, 'right corner'),
        (right_end, 'top straight'),
        (top_end, 'left corner'),
        (left_end, 'bottom straight'),
    ):
        ax.axvline(x_position, color='black', alpha=0.18, linewidth=1.0)
        ax.text(
            x_position + 0.25,
            0.5,
            label,
            rotation=90,
            alpha=0.55,
            fontsize=8,
        )

    ax.set_title('Lap Speed Profiles')
    ax.set_xlabel('Centerline progress [m]')
    ax.set_ylabel('Speed [m/s]')
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_lap_profile(
    controller,
    records,
    output_path: Path,
    field: str,
    title: str,
    ylabel: str,
    limit_lines=None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    max_lap = max(record['lap'] for record in records)
    color_map = plt.get_cmap('tab10')
    for lap in range(1, max_lap + 1):
        lap_samples = [
            (record['progress'], record[field])
            for record in records
            if record['lap'] == lap
        ]
        color = color_map((lap - 1) % 10)
        for segment_index, segment in enumerate(split_progress_segments(lap_samples)):
            x_values = [progress for progress, _ in segment]
            y_values = [value for _, value in segment]
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.25,
                label=f'Lap {lap}' if segment_index == 0 else None,
            )

    straight = controller.straight_half_length
    right_start = straight
    right_end = straight + math.pi * controller.centerline_radius
    top_end = right_end + 2.0 * straight
    left_end = top_end + math.pi * controller.centerline_radius
    for x_position, label in (
        (right_start, 'right corner'),
        (right_end, 'top straight'),
        (top_end, 'left corner'),
        (left_end, 'bottom straight'),
    ):
        ax.axvline(x_position, color='black', alpha=0.18, linewidth=1.0)
        ax.text(
            x_position + 0.25,
            0.05,
            label,
            rotation=90,
            alpha=0.55,
            fontsize=8,
        )

    if limit_lines:
        for value, color, label in limit_lines:
            ax.axhline(
                value,
                color=color,
                linestyle='--',
                alpha=0.4,
                linewidth=1.0,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel('Centerline progress [m]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_tire_usage_time(records, lap_times, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    times = [record['time'] for record in records]
    front_usage = [record['front_usage'] for record in records]
    rear_usage = [record['rear_usage'] for record in records]
    ax.plot(times, front_usage, linewidth=1.25, label='front')
    ax.plot(times, rear_usage, linewidth=1.25, label='rear')
    for time_value in lap_times:
        ax.axvline(time_value, color='black', alpha=0.12, linewidth=1.0)
    slip_times = [
        record['time'] for record in records
        if record['front_slip'] or record['rear_slip']
    ]
    if slip_times:
        ax.scatter(
            slip_times,
            [1.02] * len(slip_times),
            color='red',
            s=14,
            label='slip flag',
            zorder=5,
        )
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_title('Estimated Tire Force Usage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('force usage / static limit')
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_curvature_distance(controller, records, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)

    axes[0].plot(
        controller.reference_s,
        controller.reference_curvature,
        color='tab:blue',
        linewidth=1.35,
    )
    axes[0].set_title('Reference Curvature vs Distance')
    axes[0].set_ylabel('Reference curvature [1/m]')
    axes[0].grid(True, alpha=0.28)

    max_lap = max(record['lap'] for record in records)
    color_map = plt.get_cmap('tab10')
    for lap in range(1, max_lap + 1):
        lap_samples = [
            (record['progress'], record['actual_curvature'])
            for record in records
            if record['lap'] == lap
        ]
        color = color_map((lap - 1) % 10)
        for segment_index, segment in enumerate(split_progress_segments(lap_samples)):
            x_values = [progress for progress, _ in segment]
            y_values = [curvature for _, curvature in segment]
            axes[1].plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.0,
                label=f'Lap {lap}' if segment_index == 0 else None,
            )

    axes[1].set_title('Actual Curvature Estimate vs Distance')
    axes[1].set_xlabel('Centerline progress [m]')
    axes[1].set_ylabel('yaw_rate / speed [1/m]')
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(ncol=5, fontsize=8)

    straight = controller.straight_half_length
    right_start = straight
    right_end = straight + math.pi * controller.centerline_radius
    top_end = right_end + 2.0 * straight
    left_end = top_end + math.pi * controller.centerline_radius
    for ax in axes:
        for x_position, label in (
            (right_start, 'right corner'),
            (right_end, 'top straight'),
            (top_end, 'left corner'),
            (left_end, 'bottom straight'),
        ):
            ax.axvline(x_position, color='black', alpha=0.18, linewidth=1.0)
            ax.text(
                x_position + 0.25,
                0.02,
                label,
                rotation=90,
                alpha=0.55,
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_csv(records, output_path: Path) -> None:
    fields = [
        'time',
        'lap',
        'progress',
        'x',
        'y',
        'speed',
        'steering',
        'accel_cmd',
        'steering_rate_cmd',
        'actual_curvature',
        'surface_friction',
        'front_usage',
        'rear_usage',
        'front_slip',
        'rear_slip',
        'collision',
        'outside_track',
    ]
    with output_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def run_analysis(seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    sim = RacingSimulator()
    controller = RacingController()
    sim.random_generator.seed(seed)
    sim.started = True

    command_accel = 0.0
    command_steering_rate = 0.0
    control_period = 0.05
    next_control_time = 0.0
    sim_time = 0.0
    records = []
    lap_times = []
    lap_start_time = 0.0
    lap_summaries = []

    while sim_time < 180.0 and sim.lap_count < sim.goal_laps:
        if sim_time + 1e-9 >= next_control_time:
            command = controller.compute_control(make_state_msg(sim.state))
            command_accel = float(command.acceleration)
            command_steering_rate = float(command.steering_rate)
            next_control_time += control_period

        next_steering = sim.clamp(
            sim.state.steering + command_steering_rate * sim.dt,
            -sim.max_steering,
            sim.max_steering,
        )
        front_usage, rear_usage = tire_usage(sim, next_steering, command_accel)
        progress = controller._centerline_progress(sim.state.x, sim.state.y)
        actual_curvature = 0.0
        if abs(sim.state.speed) > 0.25:
            actual_curvature = sim.state.yaw_rate / sim.state.speed
        previous_lap = sim.lap_count

        trial_state = sim.propagate(
            sim.state,
            command_accel,
            command_steering_rate,
        )
        front_slip = bool(sim.front_tires_slipping)
        rear_slip = bool(sim.rear_tires_slipping)
        is_outside_track = sim.is_vehicle_outside_track(trial_state)
        is_in_collision = sim.is_vehicle_in_collision(trial_state)
        sim.state = trial_state
        sim.update_penalties(is_in_collision, is_outside_track)
        sim.update_waypoint_state(sim.state)

        records.append({
            'time': sim_time,
            'lap': min(previous_lap + 1, sim.goal_laps),
            'progress': progress,
            'x': sim.state.x,
            'y': sim.state.y,
            'speed': sim.state.speed,
            'steering': sim.state.steering,
            'accel_cmd': command_accel,
            'steering_rate_cmd': command_steering_rate,
            'actual_curvature': actual_curvature,
            'surface_friction': sim.surface_friction_multiplier,
            'front_usage': front_usage,
            'rear_usage': rear_usage,
            'front_slip': front_slip,
            'rear_slip': rear_slip,
            'collision': is_in_collision,
            'outside_track': is_outside_track,
        })

        if sim.lap_count > previous_lap:
            lap_times.append(sim_time)
            lap_summaries.append((
                sim.lap_count,
                sim_time - lap_start_time,
                sim.surface_friction_multiplier,
            ))
            lap_start_time = sim_time

        sim_time += sim.dt

    speed_path = output_dir / 'lap_speed_profiles.png'
    steering_rate_path = output_dir / 'steering_rate_profiles.png'
    acceleration_path = output_dir / 'acceleration_profiles.png'
    steering_path = output_dir / 'steering_angle_profiles.png'
    usage_path = output_dir / 'tire_usage_time.png'
    curvature_path = output_dir / 'distance_curvature.png'
    csv_path = output_dir / 'racing_trace.csv'
    summary_path = output_dir / 'summary.txt'

    plot_speed_profiles(controller, records, speed_path)
    plot_lap_profile(
        controller,
        records,
        steering_rate_path,
        'steering_rate_cmd',
        'Lap Steering Rate Command Profiles',
        'Steering rate command [rad/s]',
        limit_lines=[
            (controller.max_steering_rate, 'red', '+limit'),
            (-controller.max_steering_rate, 'red', '-limit'),
        ],
    )
    plot_lap_profile(
        controller,
        records,
        acceleration_path,
        'accel_cmd',
        'Lap Acceleration Command Profiles',
        'Acceleration command [m/s^2]',
        limit_lines=[
            (controller.max_accel, 'red', '+limit'),
            (-controller.max_decel, 'red', '-limit'),
        ],
    )
    plot_lap_profile(
        controller,
        records,
        steering_path,
        'steering',
        'Lap Steering Angle Profiles',
        'Steering angle [rad]',
        limit_lines=[
            (controller.max_steering, 'red', '+limit'),
            (-controller.max_steering, 'red', '-limit'),
        ],
    )
    plot_tire_usage_time(records, lap_times, usage_path)
    plot_curvature_distance(controller, records, curvature_path)
    write_csv(records, csv_path)

    max_by_lap = {}
    slip_laps = set()
    for record in records:
        lap = int(record['lap'])
        usage = max(record['front_usage'], record['rear_usage'])
        current = max_by_lap.get(lap)
        if current is None or usage > current['usage']:
            max_by_lap[lap] = {
                'usage': usage,
                'front_usage': record['front_usage'],
                'rear_usage': record['rear_usage'],
                'time': record['time'],
                'x': record['x'],
                'y': record['y'],
                'speed': record['speed'],
                'steering': record['steering'],
            }
        if record['front_slip'] or record['rear_slip']:
            slip_laps.add(lap)

    with summary_path.open('w') as summary:
        summary.write(f'seed={seed}\n')
        summary.write(
            f'elapsed={sim_time:.2f}s, '
            f'penalty={sim.total_penalty_time:.2f}s, '
            f'score={sim_time + sim.total_penalty_time:.2f}s, '
            f'collisions={sim.collision_count}, '
            f'off_track={sim.off_track_count}, '
            f'off_track_duration={sim.off_track_duration:.2f}s\n'
        )
        summary.write('lap_times_and_friction:\n')
        for lap, lap_time, friction in lap_summaries:
            summary.write(
                f'  lap {lap}: {lap_time:.2f}s, '
                f'friction={100.0 * friction:.0f}%\n'
            )
        summary.write('max_tire_usage_by_lap:\n')
        for lap in sorted(max_by_lap):
            item = max_by_lap[lap]
            summary.write(
                f'  lap {lap}: usage={item["usage"]:.3f}, '
                f'front={item["front_usage"]:.3f}, '
                f'rear={item["rear_usage"]:.3f}, '
                f't={item["time"]:.2f}s, '
                f'pos=({item["x"]:.2f},{item["y"]:.2f}), '
                f'speed={item["speed"]:.2f}, '
                f'steering={item["steering"]:.3f}\n'
            )
        summary.write(
            'slip_laps=' +
            (','.join(str(lap) for lap in sorted(slip_laps)) or 'none') +
            '\n'
        )

    print(speed_path)
    print(steering_rate_path)
    print(acceleration_path)
    print(steering_path)
    print(usage_path)
    print(curvature_path)
    print(csv_path)
    print(summary_path)

    controller.destroy_node()
    sim.destroy_node()
    rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/home/noh/ros2_ws/racing_analysis'),
    )
    parser.add_argument('--no-timestamp-dir', action='store_true')
    args = parser.parse_args()
    if not args.no_timestamp_dir:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = args.output_dir / stamp
    run_analysis(args.seed, args.output_dir)


if __name__ == '__main__':
    main()
