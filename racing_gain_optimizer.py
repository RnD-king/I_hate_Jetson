import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import rclpy

from ros2_racing_challenge.racing_controller import RacingController
from ros2_racing_challenge.racing_sim import RacingSimulator
from ros2_racing_challenge_msgs.msg import CarState


PARAMETER_BOUNDS = {
    'heading_gain': (0.8, 2.2),
    'lateral_gain': (0.8, 2.5),
    'steering_servo_gain': (5.0, 11.5),
    'speed_gain': (1.0, 2.5),
    'friction_margin': (0.94, 1.02),
    'steering_rate_margin': (0.86, 1.0),
    'min_steering_rate_speed': (5.2, 7.2),
    'next_lap_friction_preview_distance': (0.0, 8.0),
    'longitudinal_grip_reserve': (0.05, 0.45),
    'preview_brake_accel': (3.0, 5.5),
    'feedback_curvature_weight': (0.0, 0.50),
    'tire_force_usage_limit': (1.00, 1.10),
    'motion_filter_alpha': (0.35, 0.90),
    'large_lateral_error_threshold': (0.45, 0.75),
    'large_heading_error_threshold': (0.50, 0.85),
    'large_error_speed_scale': (0.35, 0.70),
    'medium_lateral_error_threshold': (0.28, 0.50),
    'medium_heading_error_threshold': (0.32, 0.58),
    'medium_error_speed_scale': (0.55, 0.82),
    'small_lateral_error_threshold': (0.14, 0.32),
    'small_heading_error_threshold': (0.18, 0.38),
    'small_error_speed_scale': (0.75, 0.94),
    'high_steering_threshold': (0.50, 0.59),
    'high_steering_speed_cap': (3.5, 4.8),
    'medium_steering_threshold': (0.42, 0.52),
    'medium_steering_speed_cap': (4.2, 5.5),
    'first_lap_start_cap_distance': (0.0, 8.0),
    'first_lap_start_speed_cap': (4.5, 7.5),
}

CORE_PARAMETERS = [
    'heading_gain',
    'lateral_gain',
    'steering_servo_gain',
    'speed_gain',
    'friction_margin',
    'next_lap_friction_preview_distance',
    'longitudinal_grip_reserve',
    'preview_brake_accel',
    'feedback_curvature_weight',
    'tire_force_usage_limit',
    'motion_filter_alpha',
]

THRESHOLD_PARAMETERS = [
    'large_lateral_error_threshold',
    'large_heading_error_threshold',
    'large_error_speed_scale',
    'medium_lateral_error_threshold',
    'medium_heading_error_threshold',
    'medium_error_speed_scale',
    'small_lateral_error_threshold',
    'small_heading_error_threshold',
    'small_error_speed_scale',
    'high_steering_threshold',
    'high_steering_speed_cap',
    'medium_steering_threshold',
    'medium_steering_speed_cap',
    'first_lap_start_cap_distance',
    'first_lap_start_speed_cap',
]


def make_state_msg(state) -> CarState:
    msg = CarState()
    msg.x = float(state.x)
    msg.y = float(state.y)
    msg.yaw = float(state.yaw)
    msg.speed = float(state.speed)
    msg.steering = float(state.steering)
    return msg


def parse_seed_list(text: str) -> List[int]:
    seeds = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            start_text, end_text = item.split(':', 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(item))
    return seeds


def selected_parameters(mode: str) -> List[str]:
    if mode == 'core':
        return list(CORE_PARAMETERS)
    if mode == 'thresholds':
        return list(THRESHOLD_PARAMETERS)
    if mode == 'all':
        return list(CORE_PARAMETERS) + list(THRESHOLD_PARAMETERS)
    params = [part.strip() for part in mode.split(',') if part.strip()]
    unknown = [param for param in params if param not in PARAMETER_BOUNDS]
    if unknown:
        raise ValueError(f'unknown parameter(s): {unknown}')
    return params


def collect_baseline(parameters: Iterable[str]) -> Dict[str, float]:
    controller = RacingController()
    baseline = {name: float(getattr(controller, name)) for name in parameters}
    controller.destroy_node()
    return baseline


def apply_parameters(controller: RacingController, params: Dict[str, float]) -> None:
    for name, value in params.items():
        setattr(controller, name, float(value))


def run_trial(
    seed: int,
    params: Dict[str, float],
    max_time: float,
    slip_penalty: float,
    failure_penalty: float,
    roughness_penalty: float,
) -> Dict[str, float]:
    sim = RacingSimulator()
    controller = RacingController()
    apply_parameters(controller, params)
    sim.random_generator.seed(seed)
    sim.started = True

    command_accel = 0.0
    command_steering_rate = 0.0
    previous_accel = 0.0
    previous_steering_rate = 0.0
    command_roughness = 0.0
    control_period = 0.05
    next_control_time = 0.0
    sim_time = 0.0
    lap_start_time = 0.0
    lap_times = []
    slip_count = 0
    max_speed = 0.0
    max_abs_steering = 0.0

    while sim_time < max_time and sim.lap_count < sim.goal_laps:
        if sim_time + 1e-9 >= next_control_time:
            command = controller.compute_control(make_state_msg(sim.state))
            command_accel = float(command.acceleration)
            command_steering_rate = float(command.steering_rate)
            command_roughness += (
                abs(command_accel - previous_accel) +
                0.15 * abs(command_steering_rate - previous_steering_rate)
            )
            previous_accel = command_accel
            previous_steering_rate = command_steering_rate
            next_control_time += control_period

        previous_lap = sim.lap_count
        trial_state = sim.propagate(
            sim.state,
            command_accel,
            command_steering_rate,
        )
        if sim.front_tires_slipping or sim.rear_tires_slipping:
            slip_count += 1

        is_outside_track = sim.is_vehicle_outside_track(trial_state)
        is_in_collision = sim.is_vehicle_in_collision(trial_state)
        sim.state = trial_state
        sim.update_penalties(is_in_collision, is_outside_track)
        sim.update_waypoint_state(sim.state)

        max_speed = max(max_speed, sim.state.speed)
        max_abs_steering = max(max_abs_steering, abs(sim.state.steering))

        if sim.lap_count > previous_lap:
            lap_times.append(sim_time - lap_start_time)
            lap_start_time = sim_time

        sim_time += sim.dt

    completed = sim.lap_count >= sim.goal_laps
    elapsed = sum(lap_times) if completed else max_time
    score = elapsed + sim.total_penalty_time
    score += slip_penalty * slip_count
    score += roughness_penalty * command_roughness
    if not completed:
        score += failure_penalty + 20.0 * (sim.goal_laps - sim.lap_count)
    score += failure_penalty * sim.collision_count
    score += 50.0 * sim.off_track_duration

    result = {
        'seed': seed,
        'score': score,
        'elapsed': elapsed,
        'penalty_time': sim.total_penalty_time,
        'slip_count': slip_count,
        'collision_count': sim.collision_count,
        'off_track_count': sim.off_track_count,
        'off_track_duration': sim.off_track_duration,
        'completed': 1.0 if completed else 0.0,
        'max_speed': max_speed,
        'max_abs_steering': max_abs_steering,
    }
    controller.destroy_node()
    sim.destroy_node()
    return result


def evaluate_candidate(
    params: Dict[str, float],
    seeds: List[int],
    max_time: float,
    slip_penalty: float,
    failure_penalty: float,
    roughness_penalty: float,
) -> Dict[str, float]:
    trial_results = [
        run_trial(
            seed,
            params,
            max_time,
            slip_penalty,
            failure_penalty,
            roughness_penalty,
        )
        for seed in seeds
    ]
    count = max(len(trial_results), 1)
    total_score = sum(result['score'] for result in trial_results)
    return {
        'objective': total_score / count,
        'mean_elapsed': sum(result['elapsed'] for result in trial_results) / count,
        'max_elapsed': max(result['elapsed'] for result in trial_results),
        'total_slips': sum(result['slip_count'] for result in trial_results),
        'total_collisions': sum(result['collision_count'] for result in trial_results),
        'total_off_track': sum(result['off_track_count'] for result in trial_results),
        'completed_trials': sum(result['completed'] for result in trial_results),
        'trial_count': float(count),
    }


def valid_threshold_order(params: Dict[str, float]) -> bool:
    lateral_ok = (
        params.get('small_lateral_error_threshold', 0.0) <
        params.get('medium_lateral_error_threshold', 1.0) <
        params.get('large_lateral_error_threshold', 2.0)
    )
    heading_ok = (
        params.get('small_heading_error_threshold', 0.0) <
        params.get('medium_heading_error_threshold', 1.0) <
        params.get('large_heading_error_threshold', 2.0)
    )
    steering_ok = (
        params.get('medium_steering_threshold', 0.0) <
        params.get('high_steering_threshold', 1.0)
    )
    scale_ok = (
        params.get('large_error_speed_scale', 0.0) <=
        params.get('medium_error_speed_scale', 1.0) <=
        params.get('small_error_speed_scale', 2.0)
    )
    cap_ok = (
        params.get('high_steering_speed_cap', 0.0) <=
        params.get('medium_steering_speed_cap', 10.0)
    )
    return lateral_ok and heading_ok and steering_ok and scale_ok and cap_ok


def sample_params(
    rng: random.Random,
    parameters: List[str],
    baseline: Dict[str, float],
    center: Optional[Dict[str, float]],
    radius: float,
) -> Dict[str, float]:
    params = dict(baseline)
    for name in parameters:
        lower, upper = PARAMETER_BOUNDS[name]
        if center is None:
            value = rng.uniform(lower, upper)
        else:
            sigma = radius * (upper - lower)
            value = rng.gauss(center[name], sigma)
        params[name] = min(upper, max(lower, value))
    return params


def write_csv(
    path: Path,
    rows: List[Dict[str, float]],
    parameters: List[str],
) -> None:
    fieldnames = [
        'iteration',
        'phase',
        'objective',
        'mean_elapsed',
        'max_elapsed',
        'total_slips',
        'total_collisions',
        'total_off_track',
        'completed_trials',
        'trial_count',
    ] + parameters
    with path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})


def optimize(args) -> None:
    parameters = selected_parameters(args.params)
    train_seeds = parse_seed_list(args.train_seeds)
    validation_seeds = parse_seed_list(args.validation_seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.random_seed)
    baseline = collect_baseline(parameters)
    best_params = dict(baseline)
    results = []
    iteration = 0

    baseline_metrics = evaluate_candidate(
        best_params,
        train_seeds,
        args.max_time,
        args.slip_penalty,
        args.failure_penalty,
        args.roughness_penalty,
    )
    best_objective = baseline_metrics['objective']
    baseline_row = {
        'iteration': iteration,
        'phase': 'baseline',
        **baseline_metrics,
        **best_params,
    }
    results.append(baseline_row)
    print(
        f'baseline objective={best_objective:.3f} '
        f'mean_elapsed={baseline_metrics["mean_elapsed"]:.3f} '
        f'slips={baseline_metrics["total_slips"]:.0f}'
    )

    phases = [('global', args.iterations, None, 1.0)]
    for round_index in range(args.local_rounds):
        radius = args.local_radius * (0.55 ** round_index)
        phases.append((f'local{round_index + 1}', args.local_samples, best_params, radius))

    for phase, sample_count, center, radius in phases:
        for _ in range(sample_count):
            iteration += 1
            candidate = sample_params(
                rng,
                parameters,
                baseline,
                center,
                radius,
            )
            if not valid_threshold_order(candidate):
                continue

            metrics = evaluate_candidate(
                candidate,
                train_seeds,
                args.max_time,
                args.slip_penalty,
                args.failure_penalty,
                args.roughness_penalty,
            )
            row = {
                'iteration': iteration,
                'phase': phase,
                **metrics,
                **candidate,
            }
            results.append(row)
            if metrics['objective'] < best_objective:
                best_objective = metrics['objective']
                best_params = dict(candidate)
                print(
                    f'new_best iter={iteration} phase={phase} '
                    f'objective={best_objective:.3f} '
                    f'mean_elapsed={metrics["mean_elapsed"]:.3f} '
                    f'slips={metrics["total_slips"]:.0f}'
                )

    validation_metrics = evaluate_candidate(
        best_params,
        validation_seeds,
        args.max_time,
        args.slip_penalty,
        args.failure_penalty,
        args.roughness_penalty,
    )

    results.sort(key=lambda row: row['objective'])
    write_csv(output_dir / 'gain_optimization_results.csv', results, parameters)
    best_payload = {
        'parameters': parameters,
        'train_seeds': train_seeds,
        'validation_seeds': validation_seeds,
        'best_params': best_params,
        'best_train_objective': best_objective,
        'validation_metrics': validation_metrics,
        'bounds': {name: PARAMETER_BOUNDS[name] for name in parameters},
    }
    with (output_dir / 'gain_optimization_best.json').open('w') as json_file:
        json.dump(best_payload, json_file, indent=2, sort_keys=True)

    print('\nTop candidates:')
    for row in results[: min(args.print_top, len(results))]:
        print(
            f'iter={row["iteration"]} phase={row["phase"]} '
            f'objective={row["objective"]:.3f} '
            f'mean_elapsed={row["mean_elapsed"]:.3f} '
            f'slips={row["total_slips"]:.0f}'
        )

    print('\nBest parameters:')
    for name in parameters:
        print(f'  {name} = {best_params[name]:.6f}')
    print(
        '\nValidation '
        f'objective={validation_metrics["objective"]:.3f} '
        f'mean_elapsed={validation_metrics["mean_elapsed"]:.3f} '
        f'slips={validation_metrics["total_slips"]:.0f}'
    )
    print(output_dir / 'gain_optimization_results.csv')
    print(output_dir / 'gain_optimization_best.json')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='core')
    parser.add_argument('--train-seeds', default='0:9')
    parser.add_argument('--validation-seeds', default='10:19')
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--local-rounds', type=int, default=2)
    parser.add_argument('--local-samples', type=int, default=40)
    parser.add_argument('--local-radius', type=float, default=0.12)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--max-time', type=float, default=180.0)
    parser.add_argument('--slip-penalty', type=float, default=0.25)
    parser.add_argument('--roughness-penalty', type=float, default=0.0005)
    parser.add_argument('--failure-penalty', type=float, default=1000.0)
    parser.add_argument('--output-dir', default='racing_gain_optimization')
    parser.add_argument('--print-top', type=int, default=8)
    args = parser.parse_args()

    rclpy.init(args=['--ros-args', '--log-level', 'fatal'])
    try:
        optimize(args)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
