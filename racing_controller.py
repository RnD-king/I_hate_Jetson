import math
from typing import List, Optional, Sequence, Tuple

from ros2_racing_challenge_msgs.msg import CarCommand, CarState
import rclpy
from rclpy.node import Node


class RacingController(Node):
    def __init__(self) -> None:
        super().__init__('racing_controller')
        self.current_state: Optional[CarState] = None

        self.drive_publisher = self.create_publisher(
            CarCommand,
            '/racing/command',
            10,
        )
        self.state_subscriber = self.create_subscription(
            CarState,
            '/racing/state',
            self.state_callback,
            10,
        )
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(
            'Racing controller ready. '
        )

        self._configure_controller()

    def state_callback(self, msg: CarState) -> None:
        self.current_state = msg

    def timer_callback(self) -> None:
        command = CarCommand()
        if self.current_state is not None:
            command = self.compute_control(
                self.current_state
            )
        self.drive_publisher.publish(command)

    def compute_control(self, state: CarState) -> CarCommand:
        """
        Return a CarCommand message.

        - `acceleration` unit: m/s^2
        - `steering_rate` unit: rad/s
        - simulator limits: acceleration in [-4.5, 2.8],
          steering_rate in [-6.0, 6.0]
        """
        self._update_progress_estimate(state)
        self._update_motion_estimate(state)

        nearest_index = self._nearest_reference_index(state)
        steering_preview = 0.55 + 0.18 * max(0.0, state.speed)
        steering_index = self._advance_index(nearest_index, steering_preview)
        center_progress = self._centerline_progress(state.x, state.y)
        direct_finish_command = self._direct_finish_sprint_command(
            state,
            center_progress,
            nearest_index,
        )
        if direct_finish_command is not None:
            return direct_finish_command

        centerline_weight = self._first_lap_centerline_weight(center_progress)

        x_ref = self.reference_x[nearest_index]
        y_ref = self.reference_y[nearest_index]
        yaw_ref = self.reference_yaw[steering_index]
        curvature_ref = self.reference_curvature[steering_index]
        yaw_nearest = self.reference_yaw[nearest_index]

        if centerline_weight > 0.0:
            center_x, center_y, center_yaw = self._centerline_pose(
                center_progress
            )
            x_ref = (
                centerline_weight * center_x +
                (1.0 - centerline_weight) * x_ref
            )
            y_ref = (
                centerline_weight * center_y +
                (1.0 - centerline_weight) * y_ref
            )
            yaw_ref = self._blend_angle(
                yaw_ref,
                center_yaw,
                centerline_weight,
            )
            yaw_nearest = self._blend_angle(
                yaw_nearest,
                center_yaw,
                centerline_weight,
            )
            curvature_ref *= (1.0 - centerline_weight)

        finish_weight = self._finish_shortcut_weight(center_progress)
        if finish_weight > 0.0:
            (
                shortcut_x,
                shortcut_y,
                shortcut_yaw,
            ) = self._finish_shortcut_pose(state.x, state.y)
            x_ref = (
                (1.0 - finish_weight) * x_ref +
                finish_weight * shortcut_x
            )
            y_ref = (
                (1.0 - finish_weight) * y_ref +
                finish_weight * shortcut_y
            )
            yaw_ref = self._blend_angle(
                yaw_ref,
                shortcut_yaw,
                finish_weight,
            )
            yaw_nearest = self._blend_angle(
                yaw_nearest,
                shortcut_yaw,
                finish_weight,
            )
            curvature_ref *= (1.0 - finish_weight)

        dx = state.x - x_ref
        dy = state.y - y_ref
        lateral_error = (
            -math.sin(yaw_nearest) * dx +
            math.cos(yaw_nearest) * dy
        )
        heading_error = self._normalize_angle(state.yaw - yaw_ref)
        path_heading_error = self._normalize_angle(state.yaw - yaw_nearest)

        delta_ff = math.atan(self.wheelbase * curvature_ref)
        speed_for_gain = max(abs(state.speed), 1.0)
        delta_feedback = (
            -self.heading_gain * heading_error -
            math.atan2(self.lateral_gain * lateral_error, speed_for_gain)
        )
        delta_target = self._clamp(
            delta_ff + delta_feedback,
            -self.max_steering,
            self.max_steering,
        )

        steering_rate = self.steering_servo_gain * (
            delta_target - state.steering
        )
        steering_rate = self._clamp(
            steering_rate,
            -self.max_steering_rate,
            self.max_steering_rate,
        )

        target_speed = self._target_speed(
            nearest_index,
            state.speed,
            lateral_error,
            path_heading_error,
            state.steering,
            delta_target,
        )
        acceleration = self._speed_control_acceleration(
            target_speed,
            state.speed,
            state.steering,
            nearest_index,
        )
        if self._should_full_accel_to_finish(center_progress):
            acceleration = self.max_accel
        acceleration, steering_rate = self._limit_tire_force_command(
            state,
            acceleration,
            steering_rate,
            nearest_index,
        )

        command = CarCommand()
        command.acceleration = float(acceleration)
        command.steering_rate = float(steering_rate)
        return command

    def _configure_controller(self) -> None:
        self.wheelbase = 1.0
        self.max_speed = 9.0
        self.max_steering = 0.60
        self.max_accel = 2.8
        self.max_decel = 4.5
        self.max_steering_rate = 6.0
        self.gravity = 9.81
        self.base_static_friction = 0.85
        self.vehicle_mass = 25.0
        self.cg_to_front_axle = 0.40
        self.cg_to_rear_axle = self.wheelbase - self.cg_to_front_axle
        self.base_front_cornering_stiffness = 260.0
        self.base_rear_cornering_stiffness = 520.0
        self.simulation_dt = 1.0 / 30.0
        self.control_period = 0.05

        self.vehicle_width = 0.65
        self.front_overhang = 0.35
        self.rear_overhang = 0.25
        self.straight_half_length = 10.0
        self.centerline_radius = 5.0
        self.track_half_width = 2.0
        self.inner_radius = self.centerline_radius - self.track_half_width
        self.outer_radius = self.centerline_radius + self.track_half_width
        self.track_length = (
            4.0 * self.straight_half_length +
            2.0 * math.pi * self.centerline_radius
        )
        self.obstacles = [
            (4.0, self.centerline_radius + 1.00, 0.5, 2.0),
            (-4.0, self.centerline_radius - 1.00, 0.5, 2.0),
        ]

        self.reference_count = 720
        self.offset_control_points = [
            -1.451770154525,
            -1.436780343267,
            -1.232660042128,
            -0.780543219479,
            -0.072609189872,
             0.416739933798,
             0.671377040241,
             0.843342658646,
             0.938766026877,
             0.947492138428,
             0.868833286677,
             0.715749974892,
             0.493606399061,
             0.325803627963,
             0.493228033751,
             0.371476079018,
            -0.049535336388,
            -0.513868773066,
            -0.904982589336,
            -1.010678096924,
            -0.714917678924,
            -0.076753009385,
             0.511763748107,
             0.926514592013,
             1.141578127485,
             1.149915963343,
             0.951081296345,
             0.543316993486,
            -0.118731111028,
            -0.900862300822,
            -1.291279591580,
            -1.415225184245,
        ]
        self.heading_gain = 1.888540
        self.lateral_gain = 1.347229
        self.steering_servo_gain = 5.947816
        self.speed_gain = 1.903588
        self.aggressive_lateral_usage_threshold = -1.0
        self.aggressive_steering_threshold = 0.32
        self.aggressive_accel_gain_scale = 1.0
        self.aggressive_decel_gain_scale = 1.0
        self.aggressive_full_accel_error = 99.0
        self.aggressive_full_decel_error = 99.0
        self.friction_margin = 1.035221
        self.steering_rate_margin = 1.0
        self.min_steering_rate_speed = 7.2
        self.min_target_speed = 2.2
        self.surface_friction_loss_per_lap_estimate = 0.03
        self.minimum_surface_multiplier = 0.85
        self.next_lap_friction_preview_distance = 1.895927
        self.longitudinal_grip_reserve = 0.091268
        self.preview_brake_accel = 5.667102
        self.feedback_curvature_weight = 0.0 # 0.209600
        self.tire_force_usage_limit = 1.073585
        self.motion_filter_alpha = 0.453072
        self.large_lateral_error_threshold = 0.886949
        self.large_heading_error_threshold = 0.974803
        self.large_error_speed_scale = 0.875553
        self.medium_lateral_error_threshold = 0.631275
        self.medium_heading_error_threshold = 0.535054
        self.medium_error_speed_scale = 0.939694
        self.small_lateral_error_threshold = 0.381249
        self.small_heading_error_threshold = 0.321522
        self.small_error_speed_scale = 0.976111
        self.high_steering_threshold = 0.569165
        self.high_steering_speed_cap = 6.698748
        self.medium_steering_threshold = 0.528223
        self.medium_steering_speed_cap = 7.615651
        self.first_lap_start_cap_distance = 0.0
        self.first_lap_start_speed_cap = 9.0
        self.low_grip_recovery_surface = 0.84 # 0.85
        self.optimized_reference_blend = 1.0
        self.outer_boundary_optimized_blend_scale = 1.0
        self.outer_boundary_blend_width = 0.42
        self.first_obstacle_optimized_blend_scale = 1.0
        self.first_obstacle_relaxation_offset = 0.14
        self.first_obstacle_relaxation_center = 31.25
        self.first_obstacle_relaxation_width = 3.2
        self.bottom_straight_relaxation_offset = 0.0
        self.bottom_straight_relaxation_center = 0.0
        self.bottom_straight_relaxation_width = 8.0
        self.bottom_straight_centering_offset = 0.10
        self.bottom_straight_centering_center = 66.0
        self.bottom_straight_centering_width = 3.2
        self.minimum_reference_offset = -1.30
        self.maximum_reference_offset = 1.45
        self.reference_offset_limit_softness = 0.08
        self.first_lap_centerline_blend_start = 6.0
        self.first_lap_centerline_blend_end = 9.0
        self.goal_laps = 10
        self.finish_shortcut_start_progress = 55.0
        self.finish_shortcut_full_progress = 56.0
        self.finish_shortcut_target_x = 0.80
        self.finish_line_target_min_y = -6.55
        self.finish_line_target_max_y = -5.75
        self.finish_direct_sprint_start_progress = 56.0
        self.finish_direct_line_x = 0.0
        self.finish_direct_min_y = -6.55
        self.finish_direct_max_y = -3.45
        self.finish_direct_max_distance = 16.0
        self.finish_direct_track_margin = 0.04
        self.finish_direct_steering_gain = 10.0
        self.finish_straight_full_accel_start = (
            self.finish_shortcut_start_progress
        )

        self.offset_second_derivatives = self._periodic_spline_second_derivatives(
            self.offset_control_points,
            self.track_length / len(self.offset_control_points),
        )
        self._build_reference_table()

        self.last_center_progress: Optional[float] = None
        self.last_lap_stamp_ns: Optional[int] = None
        self.total_progress = 0.0
        self.estimated_lap = 0
        self.last_motion_x: Optional[float] = None
        self.last_motion_y: Optional[float] = None
        self.last_motion_yaw: Optional[float] = None
        self.estimated_lateral_speed = 0.0
        self.estimated_yaw_rate = 0.0

    def _build_reference_table(self) -> None:
        self.reference_s: List[float] = []
        self.reference_x: List[float] = []
        self.reference_y: List[float] = []
        self.reference_yaw: List[float] = []
        self.reference_curvature: List[float] = []
        self.reference_steering: List[float] = []
        self.reference_d_delta_ds: List[float] = []
        self.reference_segment_length: List[float] = []
        self.reference_track_clearance: List[float] = []
        self.reference_obstacle_clearance: List[float] = []

        for index in range(self.reference_count):
            progress = self.track_length * index / self.reference_count
            center_x, center_y, center_yaw = self._centerline_pose(progress)
            offset = self._reference_offset_at(progress)
            normal_x = -math.sin(center_yaw)
            normal_y = math.cos(center_yaw)
            self.reference_s.append(progress)
            self.reference_x.append(center_x + offset * normal_x)
            self.reference_y.append(center_y + offset * normal_y)

        ds = self.track_length / self.reference_count
        yaws = []
        curvatures = []
        for index in range(self.reference_count):
            prev_index = (index - 1) % self.reference_count
            next_index = (index + 1) % self.reference_count
            dx = (
                self.reference_x[next_index] -
                self.reference_x[prev_index]
            ) / (2.0 * ds)
            dy = (
                self.reference_y[next_index] -
                self.reference_y[prev_index]
            ) / (2.0 * ds)
            ddx = (
                self.reference_x[next_index] -
                2.0 * self.reference_x[index] +
                self.reference_x[prev_index]
            ) / (ds * ds)
            ddy = (
                self.reference_y[next_index] -
                2.0 * self.reference_y[index] +
                self.reference_y[prev_index]
            ) / (ds * ds)
            denominator = max((dx * dx + dy * dy) ** 1.5, 1e-9)
            yaws.append(math.atan2(dy, dx))
            curvatures.append((dx * ddy - dy * ddx) / denominator)

        self.reference_yaw = [self._normalize_angle(yaw) for yaw in yaws]
        self.reference_curvature = curvatures
        self.reference_steering = [
            math.atan(self.wheelbase * curvature)
            for curvature in self.reference_curvature
        ]

        for index in range(self.reference_count):
            next_index = (index + 1) % self.reference_count
            segment = math.hypot(
                self.reference_x[next_index] - self.reference_x[index],
                self.reference_y[next_index] - self.reference_y[index],
            )
            self.reference_segment_length.append(max(segment, 1e-6))

        for index in range(self.reference_count):
            prev_index = (index - 1) % self.reference_count
            next_index = (index + 1) % self.reference_count
            distance = (
                self.reference_segment_length[prev_index] +
                self.reference_segment_length[index]
            )
            self.reference_d_delta_ds.append(
                (
                    self.reference_steering[next_index] -
                    self.reference_steering[prev_index]
                ) / max(distance, 1e-6)
            )

        for index in range(self.reference_count):
            corners = self._vehicle_corners_at(
                self.reference_x[index],
                self.reference_y[index],
                self.reference_yaw[index],
            )
            self.reference_track_clearance.append(
                min(self._track_clearance(x, y) for x, y in corners)
            )
            self.reference_obstacle_clearance.append(
                min(self._obstacle_clearance(x, y) for x, y in corners)
            )

    def _periodic_spline_second_derivatives(
        self,
        values: Sequence[float],
        spacing: float,
    ) -> List[float]:
        count = len(values)
        matrix = [[0.0 for _ in range(count)] for _ in range(count)]
        rhs = [0.0 for _ in range(count)]
        scale = 6.0 / (spacing * spacing)

        for row in range(count):
            matrix[row][row] = 4.0
            matrix[row][(row - 1) % count] = 1.0
            matrix[row][(row + 1) % count] = 1.0
            rhs[row] = scale * (
                values[(row + 1) % count] -
                2.0 * values[row] +
                values[(row - 1) % count]
            )

        return self._solve_linear_system(matrix, rhs)

    def _solve_linear_system(
        self,
        matrix: List[List[float]],
        rhs: List[float],
    ) -> List[float]:
        size = len(rhs)
        for pivot in range(size):
            best = max(
                range(pivot, size),
                key=lambda row: abs(matrix[row][pivot]),
            )
            if best != pivot:
                matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
                rhs[pivot], rhs[best] = rhs[best], rhs[pivot]

            pivot_value = matrix[pivot][pivot]
            if abs(pivot_value) < 1e-12:
                continue
            inv_pivot = 1.0 / pivot_value
            for col in range(pivot, size):
                matrix[pivot][col] *= inv_pivot
            rhs[pivot] *= inv_pivot

            for row in range(size):
                if row == pivot:
                    continue
                factor = matrix[row][pivot]
                if abs(factor) < 1e-12:
                    continue
                for col in range(pivot, size):
                    matrix[row][col] -= factor * matrix[pivot][col]
                rhs[row] -= factor * rhs[pivot]

        return rhs

    def _offset_at(self, progress: float) -> float:
        count = len(self.offset_control_points)
        spacing = self.track_length / count
        normalized = progress % self.track_length
        index = int(normalized / spacing) % count
        local = normalized - index * spacing
        next_index = (index + 1) % count
        a = spacing - local
        b = local
        y0 = self.offset_control_points[index]
        y1 = self.offset_control_points[next_index]
        m0 = self.offset_second_derivatives[index]
        m1 = self.offset_second_derivatives[next_index]
        return (
            m0 * a * a * a / (6.0 * spacing) +
            m1 * b * b * b / (6.0 * spacing) +
            (y0 - m0 * spacing * spacing / 6.0) * a / spacing +
            (y1 - m1 * spacing * spacing / 6.0) * b / spacing
        )

    def _reference_offset_at(self, progress: float) -> float:
        optimized_offset = self._offset_at(progress)
        safe_offset = self._smooth_limit(
            optimized_offset,
            self.minimum_reference_offset,
            self.maximum_reference_offset,
            self.reference_offset_limit_softness,
        )
        blend = self._optimized_reference_blend_at(
            progress,
            optimized_offset,
        )
        reference_offset = safe_offset + blend * (
            optimized_offset - safe_offset
        )
        reference_offset += (
            self.first_obstacle_relaxation_offset *
            self._periodic_bump(
                progress,
                self.first_obstacle_relaxation_center,
                self.first_obstacle_relaxation_width,
            )
        )
        reference_offset += (
            self.bottom_straight_relaxation_offset *
            self._periodic_bump(
                progress,
                self.bottom_straight_relaxation_center,
                self.bottom_straight_relaxation_width,
            )
        )
        reference_offset += (
            self.bottom_straight_centering_offset *
            self._periodic_bump(
                progress,
                self.bottom_straight_centering_center,
                self.bottom_straight_centering_width,
            )
        )
        return reference_offset

    def _optimized_reference_blend_at(
        self,
        progress: float,
        optimized_offset: float,
    ) -> float:
        blend = self.optimized_reference_blend

        blend *= self._outer_boundary_blend_scale(optimized_offset)

        obstacle_weight = self._periodic_bump(
            progress,
            self.first_obstacle_relaxation_center,
            1.15 * self.first_obstacle_relaxation_width,
        )
        obstacle_scale = (
            1.0 -
            (1.0 - self.first_obstacle_optimized_blend_scale) *
            obstacle_weight
        )
        return blend * obstacle_scale

    def _outer_boundary_blend_scale(self, optimized_offset: float) -> float:
        blend_start = self.minimum_reference_offset
        blend_end = blend_start + self.outer_boundary_blend_width

        if optimized_offset <= blend_start:
            return self.outer_boundary_optimized_blend_scale
        if optimized_offset >= blend_end:
            return 1.0

        ratio = (optimized_offset - blend_start) / max(
            self.outer_boundary_blend_width,
            1e-6,
        )
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        return (
            self.outer_boundary_optimized_blend_scale +
            (1.0 - self.outer_boundary_optimized_blend_scale) *
            smooth_ratio
        )

    def _periodic_bump(
        self,
        progress: float,
        center: float,
        width: float,
    ) -> float:
        distance = abs(self._progress_delta(progress, center))
        if distance >= width:
            return 0.0
        ratio = distance / width
        return 0.5 * (1.0 + math.cos(math.pi * ratio))

    def _smooth_limit(
        self,
        value: float,
        lower: float,
        upper: float,
        softness: float,
    ) -> float:
        lower_limited = (
            lower +
            softness * math.log1p(math.exp((value - lower) / softness))
        )
        return (
            upper -
            softness * math.log1p(math.exp((upper - lower_limited) / softness))
        )

    def _centerline_pose(self, progress: float) -> Tuple[float, float, float]:
        s = progress % self.track_length
        straight = self.straight_half_length
        radius = self.centerline_radius
        semicircle = math.pi * radius
        top_straight = 2.0 * straight

        if s <= straight:
            return s, -radius, 0.0

        if s <= straight + semicircle:
            arc = s - straight
            angle = -math.pi / 2.0 + arc / radius
            return (
                straight + radius * math.cos(angle),
                radius * math.sin(angle),
                self._normalize_angle(angle + math.pi / 2.0),
            )

        if s <= straight + semicircle + top_straight:
            straight_progress = s - straight - semicircle
            return straight - straight_progress, radius, math.pi

        arc = s - straight - semicircle - top_straight
        if arc <= semicircle:
            angle = math.pi / 2.0 + arc / radius
            return (
                -straight + radius * math.cos(angle),
                radius * math.sin(angle),
                self._normalize_angle(angle + math.pi / 2.0),
            )

        straight_progress = arc - semicircle
        return -straight + straight_progress, -radius, 0.0

    def _centerline_progress(self, x: float, y: float) -> float:
        straight = self.straight_half_length
        radius = self.centerline_radius

        if -straight <= x <= straight:
            if y <= 0.0:
                if x >= 0.0:
                    return x
                return (
                    3.0 * straight +
                    2.0 * math.pi * radius +
                    (x + straight)
                )
            return straight + math.pi * radius + (straight - x)

        if x > straight:
            angle = math.atan2(y, x - straight)
            if angle < -math.pi / 2.0:
                angle += 2.0 * math.pi
            return straight + radius * (angle + math.pi / 2.0)

        angle = math.atan2(y, x + straight)
        if angle < math.pi / 2.0:
            angle += 2.0 * math.pi
        return 3.0 * straight + math.pi * radius + radius * (
            angle - math.pi / 2.0
        )

    def _update_progress_estimate(self, state: CarState) -> None:
        center_progress = self._centerline_progress(state.x, state.y)
        previous_lap = self.estimated_lap
        if self.last_center_progress is not None:
            delta = self._progress_delta(
                center_progress,
                self.last_center_progress,
            )
            if delta > -0.2:
                self.total_progress += max(0.0, delta)
        self.last_center_progress = center_progress
        self.estimated_lap = int(self.total_progress / self.track_length)

        now_ns = self.get_clock().now().nanoseconds
        if self.last_lap_stamp_ns is None:
            self.last_lap_stamp_ns = now_ns

        if self.estimated_lap > previous_lap:
            lap_time = (now_ns - self.last_lap_stamp_ns) * 1e-9
            surface_multiplier = self._estimate_surface_multiplier()
            self.get_logger().info(
                f'Controller lap {self.estimated_lap} estimate: '
                f'previous_lap_time={lap_time:.2f}s, '
                f'friction={100.0 * surface_multiplier:.0f}%, '
                f'mu={self._estimate_mu():.3f}'
            )
            self.last_lap_stamp_ns = now_ns

    def _nearest_reference_index(self, state: CarState) -> int:
        center_progress = self._centerline_progress(state.x, state.y)
        center_index = int(
            self.reference_count * center_progress / self.track_length
        ) % self.reference_count
        window = 90
        best_index = center_index
        best_distance = float('inf')

        for offset in range(-window, window + 1):
            index = (center_index + offset) % self.reference_count
            dx = state.x - self.reference_x[index]
            dy = state.y - self.reference_y[index]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    def _advance_index(self, start_index: int, distance: float) -> int:
        remaining = max(0.0, distance)
        index = start_index
        while remaining > 0.0:
            remaining -= self.reference_segment_length[index]
            index = (index + 1) % self.reference_count
        return index

    def _first_lap_centerline_weight(self, center_progress: float) -> float:
        if self.estimated_lap != 0:
            return 0.0
        if self.total_progress >= self.first_lap_centerline_blend_end:
            return 0.0
        if center_progress >= self.first_lap_centerline_blend_end:
            return 0.0
        if center_progress <= self.first_lap_centerline_blend_start:
            return 1.0

        span = (
            self.first_lap_centerline_blend_end -
            self.first_lap_centerline_blend_start
        )
        ratio = (
            center_progress - self.first_lap_centerline_blend_start
        ) / max(span, 1e-6)
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        return 1.0 - smooth_ratio

    def _should_full_accel_to_finish(self, center_progress: float) -> bool:
        if self.estimated_lap < self.goal_laps - 1:
            return False
        return center_progress >= self.finish_straight_full_accel_start

    def _direct_finish_sprint_command(
        self,
        state: CarState,
        center_progress: float,
        nearest_index: int,
    ) -> Optional[CarCommand]:
        if not self._can_direct_finish_sprint(state, center_progress):
            return None

        acceleration = self.max_accel
        steering_rate = self._clamp(
            -self.finish_direct_steering_gain * state.steering,
            -self.max_steering_rate,
            self.max_steering_rate,
        )
        acceleration, steering_rate = self._limit_tire_force_command(
            state,
            acceleration,
            steering_rate,
            nearest_index,
        )

        command = CarCommand()
        command.acceleration = float(acceleration)
        command.steering_rate = float(steering_rate)
        return command

    def _can_direct_finish_sprint(
        self,
        state: CarState,
        center_progress: float,
    ) -> bool:
        if self.estimated_lap < self.goal_laps - 1:
            return False
        if center_progress < self.finish_direct_sprint_start_progress:
            return False

        cos_yaw = math.cos(state.yaw)
        if cos_yaw <= 0.15:
            return False

        distance = (self.finish_direct_line_x - state.x) / cos_yaw
        if distance <= 0.0 or distance > self.finish_direct_max_distance:
            return False

        finish_y = state.y + distance * math.sin(state.yaw)
        if (
            finish_y < self.finish_direct_min_y or
            finish_y > self.finish_direct_max_y
        ):
            return False

        return self._direct_finish_segment_is_clear(state, distance)

    def _direct_finish_segment_is_clear(
        self,
        state: CarState,
        distance: float,
    ) -> bool:
        cos_yaw = math.cos(state.yaw)
        sin_yaw = math.sin(state.yaw)
        steps = max(4, int(distance / 0.4))

        for step in range(steps + 1):
            ratio = step / max(steps, 1)
            travel = distance * ratio
            x = state.x + travel * cos_yaw
            y = state.y + travel * sin_yaw
            corners = self._vehicle_corners_at(x, y, state.yaw)

            track_clearance = min(
                self._track_clearance(corner_x, corner_y)
                for corner_x, corner_y in corners
            )
            if track_clearance < self.finish_direct_track_margin:
                return False

            obstacle_clearance = min(
                self._obstacle_clearance(corner_x, corner_y)
                for corner_x, corner_y in corners
            )
            if obstacle_clearance < 0.02:
                return False

        return True

    def _finish_shortcut_weight(self, center_progress: float) -> float:
        if self.estimated_lap < self.goal_laps - 1:
            return 0.0
        if center_progress < self.finish_shortcut_start_progress:
            return 0.0
        if center_progress >= self.finish_shortcut_full_progress:
            return 1.0

        span = (
            self.finish_shortcut_full_progress -
            self.finish_shortcut_start_progress
        )
        ratio = (
            center_progress - self.finish_shortcut_start_progress
        ) / max(span, 1e-6)
        return ratio * ratio * (3.0 - 2.0 * ratio)

    def _finish_shortcut_pose(
        self,
        vehicle_x: float,
        vehicle_y: float,
    ) -> Tuple[float, float, float]:
        start_x, start_y, _start_yaw = self._reference_pose_at_progress(
            self.finish_shortcut_start_progress,
        )
        end_x = self.finish_shortcut_target_x
        end_y = self._finish_line_target_y(vehicle_y)
        line_x = end_x - start_x
        line_y = end_y - start_y
        line_length_sq = max(line_x * line_x + line_y * line_y, 1e-9)
        projection = (
            (vehicle_x - start_x) * line_x +
            (vehicle_y - start_y) * line_y
        ) / line_length_sq
        projection = self._clamp(projection, 0.0, 1.0)
        x_ref = start_x + projection * line_x
        y_ref = start_y + projection * line_y
        yaw_ref = math.atan2(line_y, line_x)
        return x_ref, y_ref, yaw_ref

    def _finish_line_target_y(self, vehicle_y: float) -> float:
        return self._clamp(
            vehicle_y,
            self.finish_line_target_min_y,
            self.finish_line_target_max_y,
        )

    def _reference_pose_at_progress(
        self,
        progress: float,
    ) -> Tuple[float, float, float]:
        center_x, center_y, center_yaw = self._centerline_pose(progress)
        offset = self._reference_offset_at(progress)
        normal_x = -math.sin(center_yaw)
        normal_y = math.cos(center_yaw)
        return (
            center_x + offset * normal_x,
            center_y + offset * normal_y,
            center_yaw,
        )

    def _blend_angle(
        self,
        base_angle: float,
        target_angle: float,
        target_weight: float,
    ) -> float:
        delta = self._normalize_angle(target_angle - base_angle)
        return self._normalize_angle(base_angle + target_weight * delta)

    def _update_motion_estimate(self, state: CarState) -> None:
        if self.last_motion_x is None:
            self.last_motion_x = state.x
            self.last_motion_y = state.y
            self.last_motion_yaw = state.yaw
            self.estimated_lateral_speed = 0.0
            self.estimated_yaw_rate = (
                state.speed * math.tan(state.steering) / self.wheelbase
            )
            return

        dt = self.control_period
        dx = state.x - self.last_motion_x
        dy = state.y - self.last_motion_y
        dyaw = self._normalize_angle(state.yaw - self.last_motion_yaw)
        velocity_x = dx / dt
        velocity_y = dy / dt
        measured_lateral_speed = (
            -math.sin(state.yaw) * velocity_x +
            math.cos(state.yaw) * velocity_y
        )
        measured_yaw_rate = dyaw / dt
        alpha = self.motion_filter_alpha
        self.estimated_lateral_speed = self._clamp(
            alpha * measured_lateral_speed +
            (1.0 - alpha) * self.estimated_lateral_speed,
            -3.0,
            3.0,
        )
        self.estimated_yaw_rate = self._clamp(
            alpha * measured_yaw_rate +
            (1.0 - alpha) * self.estimated_yaw_rate,
            -6.0,
            6.0,
        )
        self.last_motion_x = state.x
        self.last_motion_y = state.y
        self.last_motion_yaw = state.yaw

    def _limit_tire_force_command(
        self,
        state: CarState,
        acceleration: float,
        steering_rate: float,
        nearest_index: int,
    ) -> Tuple[float, float]:
        surface_multiplier = self._estimate_surface_multiplier_for_progress(
            self.reference_s[nearest_index],
        )
        predicted_steering = self._clamp(
            state.steering + steering_rate * self.simulation_dt,
            -self.max_steering,
            self.max_steering,
        )
        acceleration = self._limit_accel_by_tire_force(
            state,
            predicted_steering,
            acceleration,
            surface_multiplier,
        )
        steering_rate = self._limit_steering_rate_by_tire_force(
            state,
            acceleration,
            steering_rate,
            surface_multiplier,
        )
        return acceleration, steering_rate

    def _limit_accel_by_tire_force(
        self,
        state: CarState,
        steering: float,
        acceleration: float,
        surface_multiplier: float,
    ) -> float:
        (
            front_fx,
            front_fy,
            rear_fx,
            rear_fy,
            front_limit,
            rear_limit,
        ) = self._estimated_tire_forces(
            state,
            steering,
            acceleration,
            surface_multiplier,
        )
        del front_fx, rear_fx
        front_load, rear_load = self._normal_loads()
        total_load = front_load + rear_load
        allowed_front_force = self._remaining_longitudinal_force(
            front_fy,
            front_limit,
        )
        allowed_rear_force = self._remaining_longitudinal_force(
            rear_fy,
            rear_limit,
        )
        allowed_accel_front = (
            allowed_front_force * total_load /
            max(self.vehicle_mass * front_load, 1e-6)
        )
        allowed_accel_rear = (
            allowed_rear_force * total_load /
            max(self.vehicle_mass * rear_load, 1e-6)
        )
        allowed_accel = min(
            abs(acceleration),
            allowed_accel_front,
            allowed_accel_rear,
            self.max_accel if acceleration >= 0.0 else self.max_decel,
        )
        if acceleration < 0.0:
            return -allowed_accel
        return allowed_accel

    def _limit_steering_rate_by_tire_force(
        self,
        state: CarState,
        acceleration: float,
        steering_rate: float,
        surface_multiplier: float,
    ) -> float:
        predicted_steering = self._clamp(
            state.steering + steering_rate * self.simulation_dt,
            -self.max_steering,
            self.max_steering,
        )
        (
            front_fx,
            _front_fy,
            _rear_fx,
            _rear_fy,
            front_limit,
            _rear_limit,
        ) = self._estimated_tire_forces(
            state,
            predicted_steering,
            acceleration,
            surface_multiplier,
        )
        lateral_capacity = math.sqrt(
            max(front_limit * front_limit - front_fx * front_fx, 0.0)
        )
        front_stiffness = (
            self.base_front_cornering_stiffness * surface_multiplier
        )
        max_front_slip = lateral_capacity / max(front_stiffness, 1e-6)
        body_front_angle = math.atan2(
            self.estimated_lateral_speed +
            self.cg_to_front_axle * self.estimated_yaw_rate,
            max(state.speed, 0.25),
        )
        requested_front_slip = predicted_steering - body_front_angle
        limited_front_slip = self._clamp(
            requested_front_slip,
            -max_front_slip,
            max_front_slip,
        )
        limited_steering = self._clamp(
            body_front_angle + limited_front_slip,
            -self.max_steering,
            self.max_steering,
        )
        limited_rate = (
            limited_steering - state.steering
        ) / self.simulation_dt
        return self._clamp(
            limited_rate,
            -self.max_steering_rate,
            self.max_steering_rate,
        )

    def _estimated_tire_forces(
        self,
        state: CarState,
        steering: float,
        acceleration: float,
        surface_multiplier: float,
    ) -> Tuple[float, float, float, float, float, float]:
        safe_speed = max(state.speed, 0.25)
        front_slip = steering - math.atan2(
            self.estimated_lateral_speed +
            self.cg_to_front_axle * self.estimated_yaw_rate,
            safe_speed,
        )
        rear_slip = -math.atan2(
            self.estimated_lateral_speed -
            self.cg_to_rear_axle * self.estimated_yaw_rate,
            safe_speed,
        )
        front_load, rear_load = self._normal_loads()
        total_load = front_load + rear_load
        longitudinal_force = self.vehicle_mass * acceleration
        front_fx = longitudinal_force * front_load / total_load
        rear_fx = longitudinal_force * rear_load / total_load
        front_fy = (
            self.base_front_cornering_stiffness *
            surface_multiplier *
            front_slip
        )
        rear_fy = (
            self.base_rear_cornering_stiffness *
            surface_multiplier *
            rear_slip
        )
        static_mu = self.base_static_friction * surface_multiplier
        front_limit = (
            self.tire_force_usage_limit * static_mu * front_load
        )
        rear_limit = self.tire_force_usage_limit * static_mu * rear_load
        return (
            front_fx,
            front_fy,
            rear_fx,
            rear_fy,
            front_limit,
            rear_limit,
        )

    def _normal_loads(self) -> Tuple[float, float]:
        front_load = (
            self.vehicle_mass * self.gravity *
            self.cg_to_rear_axle / self.wheelbase
        )
        rear_load = (
            self.vehicle_mass * self.gravity *
            self.cg_to_front_axle / self.wheelbase
        )
        return front_load, rear_load

    def _remaining_longitudinal_force(
        self,
        lateral_force: float,
        force_limit: float,
    ) -> float:
        return math.sqrt(
            max(force_limit * force_limit - lateral_force * lateral_force, 0.0)
        )

    def _target_speed(
        self,
        nearest_index: int,
        current_speed: float,
        lateral_error: float,
        heading_error: float,
        current_steering: float,
        target_steering: float,
    ) -> float:
        preview_distance = 5.0 + 0.80 * max(0.0, current_speed)
        max_curvature = 1e-4
        max_delta_rate_per_meter = 1e-4
        min_track_clearance = float('inf')
        min_obstacle_clearance = float('inf')
        surface_multiplier = self._estimate_surface_multiplier_for_progress(
            self.reference_s[nearest_index],
        )
        mu_estimate = self.base_static_friction * surface_multiplier
        lateral_accel_limit = self._usable_lateral_accel(mu_estimate)
        grip_speed_cap = self.max_speed
        distance = 0.0
        index = nearest_index

        while distance < preview_distance:
            curvature = max(abs(self.reference_curvature[index]), 1e-4)
            max_curvature = max(max_curvature, curvature)
            local_curve_speed = math.sqrt(lateral_accel_limit / curvature)
            braking_speed_cap = math.sqrt(
                local_curve_speed * local_curve_speed +
                2.0 * self.preview_brake_accel * distance
            )
            grip_speed_cap = min(grip_speed_cap, braking_speed_cap)
            max_delta_rate_per_meter = max(
                max_delta_rate_per_meter,
                abs(self.reference_d_delta_ds[index]),
            )
            min_track_clearance = min(
                min_track_clearance,
                self.reference_track_clearance[index],
            )
            min_obstacle_clearance = min(
                min_obstacle_clearance,
                self.reference_obstacle_clearance[index],
            )
            distance += self.reference_segment_length[index]
            index = (index + 1) % self.reference_count

        curve_speed = math.sqrt(lateral_accel_limit / max_curvature)
        recovery_speed = self.max_speed
        if surface_multiplier <= self.low_grip_recovery_surface:
            control_curvature = abs(math.tan(target_steering)) / self.wheelbase
            recovery_curvature = max_curvature + (
                self.feedback_curvature_weight *
                max(0.0, control_curvature - max_curvature)
            )
            recovery_speed = math.sqrt(
                lateral_accel_limit / max(recovery_curvature, 1e-4)
            )
        steering_rate_speed = (
            self.max_steering_rate *
            self.steering_rate_margin /
            max_delta_rate_per_meter
        )
        steering_rate_speed = max(
            self.min_steering_rate_speed,
            steering_rate_speed,
        )
        target_speed = min(
            self.max_speed,
            grip_speed_cap,
            curve_speed,
            recovery_speed,
            steering_rate_speed,
            self._clearance_speed_cap(
                min_track_clearance,
                min_obstacle_clearance,
            ),
        )

        abs_lateral_error = abs(lateral_error)
        abs_heading_error = abs(heading_error)
        if (
            abs_lateral_error > self.large_lateral_error_threshold or
            abs_heading_error > self.large_heading_error_threshold
        ):
            target_speed *= self.large_error_speed_scale
        elif (
            abs_lateral_error > self.medium_lateral_error_threshold or
            abs_heading_error > self.medium_heading_error_threshold
        ):
            target_speed *= self.medium_error_speed_scale
        elif (
            abs_lateral_error > self.small_lateral_error_threshold or
            abs_heading_error > self.small_heading_error_threshold
        ):
            target_speed *= self.small_error_speed_scale

        abs_steering = abs(current_steering)
        if abs_steering > self.high_steering_threshold:
            target_speed = min(target_speed, self.high_steering_speed_cap)
        elif abs_steering > self.medium_steering_threshold:
            target_speed = min(target_speed, self.medium_steering_speed_cap)

        if (
            self.estimated_lap == 0 and
            self.total_progress < self.first_lap_start_cap_distance
        ):
            target_speed = min(target_speed, self.first_lap_start_speed_cap)

        return self._clamp(
            target_speed,
            self.min_target_speed,
            self.max_speed,
        )

    def _speed_control_acceleration(
        self,
        target_speed: float,
        current_speed: float,
        current_steering: float,
        nearest_index: int,
    ) -> float:
        speed_error = target_speed - current_speed
        gain = self.speed_gain

        if self._can_use_aggressive_longitudinal_control(
            current_speed,
            current_steering,
            nearest_index,
        ):
            if speed_error >= self.aggressive_full_accel_error:
                return self.max_accel
            if speed_error <= -self.aggressive_full_decel_error:
                return -self.max_decel
            if speed_error >= 0.0:
                gain *= self.aggressive_accel_gain_scale
            else:
                gain *= self.aggressive_decel_gain_scale

        return self._clamp(
            gain * speed_error,
            -self.max_decel,
            self.max_accel,
        )

    def _can_use_aggressive_longitudinal_control(
        self,
        current_speed: float,
        current_steering: float,
        nearest_index: int,
    ) -> bool:
        if abs(current_steering) > self.aggressive_steering_threshold:
            return False

        surface_multiplier = self._estimate_surface_multiplier_for_progress(
            self.reference_s[nearest_index],
        )
        mu_estimate = self.base_static_friction * surface_multiplier
        lateral_limit = self._usable_lateral_accel(mu_estimate)
        lateral_accel = (
            current_speed * current_speed *
            abs(self.reference_curvature[nearest_index])
        )
        lateral_usage = lateral_accel / max(lateral_limit, 1e-6)
        return lateral_usage < self.aggressive_lateral_usage_threshold

    def _estimate_mu(self) -> float:
        surface_multiplier = self._estimate_surface_multiplier()
        return self.base_static_friction * surface_multiplier

    def _estimate_surface_multiplier(self) -> float:
        return self._estimate_surface_multiplier_for_lap(self.estimated_lap)

    def _estimate_surface_multiplier_for_lap(self, lap: int) -> float:
        return max(
            self.minimum_surface_multiplier,
            1.0 - self.surface_friction_loss_per_lap_estimate *
            lap,
        )

    def _estimate_surface_multiplier_for_progress(
        self,
        progress: float,
    ) -> float:
        current_surface = self._estimate_surface_multiplier_for_lap(
            self.estimated_lap,
        )
        next_surface = self._estimate_surface_multiplier_for_lap(
            self.estimated_lap + 1,
        )
        distance_to_start = (
            self.track_length - progress % self.track_length
        ) % self.track_length

        if distance_to_start >= self.next_lap_friction_preview_distance:
            return current_surface

        ratio = 1.0 - (
            distance_to_start /
            max(self.next_lap_friction_preview_distance, 1e-6)
        )
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        return (
            (1.0 - smooth_ratio) * current_surface +
            smooth_ratio * next_surface
        )

    def _usable_lateral_accel(self, mu_estimate: float) -> float:
        grip_accel = mu_estimate * self.gravity * self.friction_margin
        reserve = min(self.longitudinal_grip_reserve, 0.45 * grip_accel)
        return math.sqrt(max(grip_accel * grip_accel - reserve * reserve, 0.1))

    def _clearance_speed_cap(
        self,
        track_clearance: float,
        obstacle_clearance: float,
    ) -> float:
        del track_clearance, obstacle_clearance
        return self.max_speed

    def _vehicle_corners_at(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> List[Tuple[float, float]]:
        half_width = 0.5 * self.vehicle_width
        front_x = self.wheelbase + self.front_overhang
        rear_x = -self.rear_overhang
        local_corners = [
            (front_x, half_width),
            (front_x, -half_width),
            (rear_x, -half_width),
            (rear_x, half_width),
        ]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return [
            (
                x + local_x * cos_yaw - local_y * sin_yaw,
                y + local_x * sin_yaw + local_y * cos_yaw,
            )
            for local_x, local_y in local_corners
        ]

    def _track_clearance(self, x: float, y: float) -> float:
        if -self.straight_half_length <= x <= self.straight_half_length:
            radial_distance = abs(y)
        elif x > self.straight_half_length:
            radial_distance = math.hypot(x - self.straight_half_length, y)
        else:
            radial_distance = math.hypot(x + self.straight_half_length, y)

        outer_clearance = self.outer_radius - radial_distance
        inner_clearance = radial_distance - self.inner_radius
        return min(outer_clearance, inner_clearance)

    def _obstacle_clearance(self, x: float, y: float) -> float:
        clearance = float('inf')
        for center_x, center_y, size_x, size_y in self.obstacles:
            dx = abs(x - center_x) - 0.5 * size_x
            dy = abs(y - center_y) - 0.5 * size_y
            clearance = min(clearance, max(dx, dy))
        return clearance

    def _progress_delta(self, current: float, previous: float) -> float:
        return (
            current - previous + 0.5 * self.track_length
        ) % self.track_length - 0.5 * self.track_length

    def _normalize_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RacingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
