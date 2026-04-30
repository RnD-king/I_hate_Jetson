#include "agent.h"

#include <cmath>
#include <limits>

using namespace std::chrono_literals;
namespace apf {
// DEBUG_TIMING_START: remove this block after measuring convergence time.
namespace {
constexpr bool DEBUG_TIMING = true;
constexpr double DEBUG_TIMING_GOAL_DIST = 0.10;
constexpr double DEBUG_TIMING_SPEED = 0.10;
} // namespace
// DEBUG_TIMING_END

ApfAgent::ApfAgent() : rclcpp::Node("agent") {
  // Agent id
  this->declare_parameter("agent_id", 0);
  agent_id = this->get_parameter("agent_id").as_int();

  // Mission file name
  this->declare_parameter("mission_file_name",
                          "~/ros2_ws/src/ros2_artificial_potential_field/"
                          "mission/mission_single_agent.yaml");
  std::string mission_file_name =
      this->get_parameter("mission_file_name").as_string();

  // Mission
  YAML::Node mission = YAML::LoadFile(mission_file_name);
  auto agents_yaml = mission["agents"];
  number_of_agents = agents_yaml.size();
  agent_positions.resize(number_of_agents);
  for (size_t id = 0; id < number_of_agents; id++) {
    agent_positions[id] = Vector3d(agents_yaml[id]["start"][0].as<double>(),
                                   agents_yaml[id]["start"][1].as<double>(),
                                   agents_yaml[id]["start"][2].as<double>());
  }
  start = agent_positions[agent_id];
  goal = Vector3d(agents_yaml[agent_id]["goal"][0].as<double>(),
                  agents_yaml[agent_id]["goal"][1].as<double>(),
                  agents_yaml[agent_id]["goal"][2].as<double>());

  auto obstacles_yaml = mission["obstacles"];
  number_of_obstacles = obstacles_yaml.size();
  obstacles.resize(number_of_obstacles);
  for (size_t obs_id = 0; obs_id < number_of_obstacles; obs_id++) {
    obstacles[obs_id].position =
        Vector3d(obstacles_yaml[obs_id]["position"][0].as<double>(),
                 obstacles_yaml[obs_id]["position"][1].as<double>(),
                 obstacles_yaml[obs_id]["position"][2].as<double>());
    obstacles[obs_id].radius = obstacles_yaml[obs_id]["radius"].as<double>();
  }

  // Initialize agent's state
  state.position = start;
  state.velocity = Vector3d(0, 0, 0);
  // DEBUG_TIMING_START: start timing from node initialization.
  debug_start_time = this->get_clock()->now();
  // DEBUG_TIMING_END

  // TF2_ROS
  tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // ROS timer
  int timer_period_ms = static_cast<int>(dt * 1000);
  timer_tf =
      this->create_wall_timer(std::chrono::milliseconds(timer_period_ms),
                              std::bind(&ApfAgent::timer_tf_callback, this));
  timer_pub = this->create_wall_timer(
      40ms, std::bind(&ApfAgent::timer_pub_callback, this));

  // ROS publisher
  pub_pose = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "robot/pose", 10);
  pub_motion_metrics = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/motion_metrics", 10);
  if (agent_id == 0) {
    pub_distance_metrics =
        this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/distance_metrics", 10);
  }
  std::cout << "[ApfAgent] Agent" << agent_id << " is ready." << std::endl;
}

void ApfAgent::timer_tf_callback() {
  listen_tf();
  update_state();
  broadcast_tf();
}

void ApfAgent::timer_pub_callback() {
  publish_marker_pose();
  publish_distance_metrics();
  publish_motion_metrics();
}

void ApfAgent::listen_tf() {
  for (size_t id = 0; id < number_of_agents; id++) {
    if (id == agent_id) { // 내꺼 저장
      agent_positions[id] = state.position;
      continue;
    }

    std::string child_frame_id = "agent" + std::to_string(id);
    try { // 다른 에이전트들 읽어오기
      geometry_msgs::msg::TransformStamped transform =
          tf_buffer->lookupTransform(frame_id, child_frame_id,
                                     tf2::TimePointZero);
      agent_positions[id] = Vector3d(transform.transform.translation.x,
                                     transform.transform.translation.y,
                                     transform.transform.translation.z);
    } catch (const tf2::TransformException &ex) { // 혹시나 실패
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Could not transform %s to %s: %s",
                           child_frame_id.c_str(), frame_id.c_str(), ex.what());
    }
  }

  // Collision check
  double min_dist = 1000000;
  for (size_t id = 0; id < number_of_agents; id++) {
    double dist = (agent_positions[id] - state.position).norm();
    if (id != agent_id and dist < min_dist) {
      min_dist = dist;
    }
  }
  if (min_dist < 2 * radius) {
    std::cout << "Collision! Minimum distance between agents: " +
                     std::to_string(min_dist)
              << std::endl;
  }

  for (size_t obs_id = 0; obs_id < number_of_obstacles; obs_id++) {
    double dist = (obstacles[obs_id].position - state.position).norm();
    if (dist < radius + obstacles[obs_id].radius) {
      std::cout << "Collision! Minimum distance between agent and obstacle: " +
                       std::to_string(dist)
                << std::endl;
    }
  }
}

void ApfAgent::update_state() {
  Vector3d u = apf_controller();
  last_control_input = u;
  state.position = state.position + state.velocity * dt + 0.5 * u * dt * dt;
  state.velocity = state.velocity + u * dt;
  agent_positions[agent_id] = state.position;

  // DEBUG_TIMING_START: print each agent's first goal-convergence time once.
  if (DEBUG_TIMING && !debug_arrival_reported) {
    Vector3d goal_error = goal - state.position;
    goal_error.z() = 0.0;
    Vector3d velocity = state.velocity;
    velocity.z() = 0.0;
    if (goal_error.norm() < DEBUG_TIMING_GOAL_DIST &&
        velocity.norm() < DEBUG_TIMING_SPEED) {
      double elapsed_time =
          (this->get_clock()->now() - debug_start_time).seconds();
      RCLCPP_INFO(this->get_logger(),
                  "[Timing] Agent%zu converged in %.3f s "
                  "(goal_dist=%.3f, speed=%.3f)",
                  agent_id, elapsed_time, goal_error.norm(), velocity.norm());
      debug_arrival_reported = true;
    }
  }
  // DEBUG_TIMING_END
}

void ApfAgent::broadcast_tf() {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = this->get_clock()->now();
  transform.header.frame_id = frame_id;
  transform.child_frame_id = "agent" + std::to_string(agent_id);

  transform.transform.translation.x = state.position.x();
  transform.transform.translation.y = state.position.y();
  transform.transform.translation.z = state.position.z();

  transform.transform.rotation.x = 0.0;
  transform.transform.rotation.y = 0.0;
  transform.transform.rotation.z = 0.0;
  transform.transform.rotation.w = 1.0;

  tf_broadcaster->sendTransform(transform);
}

Vector3d ApfAgent::apf_controller() {
  Vector3d u = Vector3d::Zero();

  // 제한값
  const double max_vel = 1.0;   // 속도
  const double max_accel = 3.0; // 가속도
  // 목표
  const double k_goal = 2.5;           // 기본 인력
  const double goal_switch_dist = 3.5; // 이 거리보다 멀면 u_goal 크기 제한
  const double k_goal_near = 4.4; // 목표점 근처에서 사용하는 인력
  const double goal_settle_dist = 1.2; // 목표점 근처 거리

  // 장애물
  const double k_obs = 8.0;              // 수직 방향 회피 척력
  const double k_obs_barrier = 1.35;     // 거리 방향 회피 척력
  const double obs_barrier_dist = 0.55;  // 척력 안전 거리
  const double obs_lookahead_time = 3.0; // 예측 충돌 시간
  const double obs_safety_margin = 0.35; // 예측 충돌 판정 안전 거리

  // 로봇
  const double k_agent_tangent = 2.5;      // 수직 방향 회피 척력
  const double k_agent = 0.08;             // 거리 방향 회피 척력
  const double k_agent_brake = 5.0;        // 거리 방향 감속
  const double agent_influence_dist = 1.2; // 척력 안전 거리
  const double agent_lookahead_time = 2.0; // 예측 충돌 시간
  const double agent_safety_margin = 0.2;  // 예측 충돌 판정 안전 거리

  // 회전
  const double k_orbit = 2.3;          // 회전 성분 게인
  const double orbit_min_radius = 0.0; // 회전 최소 반지름
  const double orbit_max_radius = 4.0; // 회전 최대 반지름
  const double orbit_goal_cutoff_dist = 3.0; // 목표점 근처 도달하면 회전 제거

  // 댐핑
  const double k_damp = 2.2;      // 기본 댐핑
  const double k_damp_near = 4.2; // 목표점 근처 댐핑

  // 기타
  const double min_prediction_speed = 0.5; // 최소 예측 충돌 속도
  const double eps = 1e-6;                 // 분모 0 방지

  // Attraction force
  Vector3d goal_error = goal - state.position;
  goal_error.z() = 0.0;
  double goal_dist = goal_error.norm();
  double settle_weight = 0.0;
  if (goal_dist < goal_settle_dist) {
    settle_weight = (goal_settle_dist - goal_dist) / goal_settle_dist;
  }
  double k_goal_active = k_goal + (k_goal_near - k_goal) * settle_weight;
  double k_damp_active = k_damp + (k_damp_near - k_damp) * settle_weight;

  Vector3d u_goal = Vector3d::Zero();
  if (goal_dist > eps) {
    if (goal_dist > goal_switch_dist) {
      u_goal = k_goal_active * goal_switch_dist * goal_error / goal_dist;
    } else {
      u_goal = k_goal_active * goal_error;
    }
  }

  // Repulsion force
  Vector3d u_obs = Vector3d::Zero();
  for (size_t obs_id = 0; obs_id < number_of_obstacles; obs_id++) {
    Vector3d to_obs = obstacles[obs_id].position - state.position;
    to_obs.z() = 0.0;
    double center_dist = to_obs.norm();
    if (center_dist < eps) {
      continue;
    }

    Vector3d away_from_obs = -to_obs / center_dist;
    double surface_dist = center_dist - radius - obstacles[obs_id].radius;
    if (surface_dist < obs_barrier_dist) {
      double rho = surface_dist;
      if (rho < eps) {
        rho = eps;
      }
      u_obs += k_obs_barrier * (1.0 / rho - 1.0 / obs_barrier_dist) /
               (rho * rho) * away_from_obs;
    }

    Vector3d heading = state.velocity;
    heading.z() = 0.0;
    double speed = heading.norm();
    if (speed > eps) {
      heading = heading / speed;
    } else if (goal_dist > eps) {
      heading = goal_error / goal_dist;
    } else {
      continue;
    }

    double forward_dist = to_obs.dot(heading);
    if (forward_dist <= 0.0) {
      continue;
    }

    double prediction_speed = speed;
    if (prediction_speed < min_prediction_speed) {
      prediction_speed = min_prediction_speed;
    }
    double time_to_closest = forward_dist / prediction_speed;
    Vector3d closest_offset = to_obs - forward_dist * heading;
    double lateral_dist = closest_offset.norm();
    double collision_radius =
        radius + obstacles[obs_id].radius + obs_safety_margin;

    if (time_to_closest < obs_lookahead_time &&
        lateral_dist < collision_radius) {
      Vector3d tangent(to_obs.y(), -to_obs.x(), 0.0);
      double tangent_norm = tangent.norm();
      if (tangent_norm > eps) {
        tangent = tangent / tangent_norm;
      } else {
        tangent = Vector3d(heading.y(), -heading.x(), 0.0);
      }

      double clearance_weight =
          (collision_radius - lateral_dist) / collision_radius;
      double time_weight =
          (obs_lookahead_time - time_to_closest) / obs_lookahead_time;
      u_obs += k_obs * clearance_weight * time_weight * tangent;
    }
  }

  Vector3d u_agent = Vector3d::Zero();
  for (size_t id = 0; id < number_of_agents; id++) {
    if (id == agent_id) {
      continue;
    }

    Vector3d to_agent = agent_positions[id] - state.position;
    to_agent.z() = 0.0;
    double center_dist = to_agent.norm();
    if (center_dist < eps) {
      continue;
    }

    Vector3d direction = -to_agent / center_dist;

    double surface_dist = center_dist - 2.0 * radius;
    if (surface_dist < agent_influence_dist) {
      double rho = surface_dist;
      if (rho < eps) {
        rho = eps;
      }
      u_agent += k_agent * (1.0 / rho - 1.0 / agent_influence_dist) /
                 (rho * rho) * direction;
    }

    Vector3d heading = state.velocity;
    heading.z() = 0.0;
    double speed = heading.norm();
    if (speed > eps) {
      heading = heading / speed;
    } else if (goal_dist > eps) {
      heading = goal_error / goal_dist;
    } else {
      continue;
    }

    double forward_dist = to_agent.dot(heading);
    if (forward_dist <= 0.0) {
      continue;
    }

    double prediction_speed = speed;
    if (prediction_speed < min_prediction_speed) {
      prediction_speed = min_prediction_speed;
    }
    double time_to_closest = forward_dist / prediction_speed;
    Vector3d closest_offset = to_agent - forward_dist * heading;
    double lateral_dist = closest_offset.norm();
    double collision_radius = 2.0 * radius + agent_safety_margin;

    if (time_to_closest < agent_lookahead_time &&
        lateral_dist < collision_radius) {
      Vector3d tangent(to_agent.y(), -to_agent.x(), 0.0);
      double tangent_norm = tangent.norm();
      if (tangent_norm > eps) {
        tangent = tangent / tangent_norm;
      } else {
        tangent = Vector3d(heading.y(), -heading.x(), 0.0);
      }

      double clearance_weight =
          (collision_radius - lateral_dist) / collision_radius;
      double time_weight =
          (agent_lookahead_time - time_to_closest) / agent_lookahead_time;
      u_agent += k_agent_tangent * clearance_weight * time_weight * tangent;
      u_agent += -k_agent_brake * clearance_weight * time_weight * heading;
    }
  }

  Vector3d center_vec = state.position;
  center_vec.z() = 0.0;
  double center_radius = center_vec.norm();
  if (center_radius > orbit_min_radius && goal_dist > orbit_goal_cutoff_dist) {
    double orbit_weight = 0.35 + 0.65 * (center_radius - orbit_min_radius) /
                                     (orbit_max_radius - orbit_min_radius);
    if (orbit_weight > 1.0) {
      orbit_weight = 1.0;
    } else if (orbit_weight < 0.0) {
      orbit_weight = 0.0;
    }

    double goal_weight =
        (goal_dist - orbit_goal_cutoff_dist) / orbit_goal_cutoff_dist;
    if (goal_weight > 1.0) {
      goal_weight = 1.0;
    } else if (goal_weight < 0.0) {
      goal_weight = 0.0;
    }

    Vector3d goal_direction = goal_error / goal_dist;
    Vector3d orbit_direction(-goal_direction.y(), goal_direction.x(), 0.0);
    u_agent += k_orbit * orbit_weight * goal_weight * orbit_direction;
  }

  // Damping force
  Vector3d u_damp = -k_damp_active * state.velocity;
  u_damp.z() = 0.0;

  // Net force
  u = u_goal + u_obs + u_agent + u_damp;
  u.z() = 0.0;

  // Limit the acceleration command by vector norm, preserving APF direction.
  double acc_norm = u.norm();
  if (acc_norm > max_accel && acc_norm > eps) {
    u = max_accel * u / acc_norm;
  }

  // Limit speed by modifying the acceleration command, not the state velocity.
  Vector3d current_velocity = state.velocity;
  current_velocity.z() = 0.0;
  Vector3d v_next = current_velocity + u * dt;
  v_next.z() = 0.0;
  double v_next_norm = v_next.norm();
  if (v_next_norm > max_vel && v_next_norm > eps) {
    Vector3d v_limited = max_vel * v_next / v_next_norm;
    u = (v_limited - current_velocity) / dt;
    u.z() = 0.0;

    acc_norm = u.norm();
    if (acc_norm > max_accel && acc_norm > eps) {
      u = max_accel * u / acc_norm;
    }
  }

  // Clamping for maximum acceleration constraint
  for (int i = 0; i < 3; i++) {
    if (u(i) > max_acc) {
      u(i) = max_acc;
    } else if (u(i) < -max_acc) {
      u(i) = -max_acc;
    }
  }

  return u;
}

void ApfAgent::publish_marker_pose() {
  // Only agent_id
  if (agent_id != 0) {
    return;
  }

  visualization_msgs::msg::MarkerArray msg;
  for (size_t id = 0; id < number_of_agents; id++) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "agent";
    marker.id = (int)id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = agent_positions[id].x();
    marker.pose.position.y = agent_positions[id].y();
    marker.pose.position.z = agent_positions[id].z();
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.scale.x = 2 * radius;
    marker.scale.y = 2 * radius;
    marker.scale.z = 2 * radius;
    marker.color.r = 0;
    marker.color.g = 0;
    marker.color.b = 1;
    marker.color.a = 0.3;
    msg.markers.emplace_back(marker);
  }

  for (size_t obs_id = 0; obs_id < number_of_obstacles; obs_id++) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "obstacle";
    marker.id = (int)obs_id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = obstacles[obs_id].position.x();
    marker.pose.position.y = obstacles[obs_id].position.y();
    marker.pose.position.z = obstacles[obs_id].position.z();
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.scale.x = 2 * obstacles[obs_id].radius;
    marker.scale.y = 2 * obstacles[obs_id].radius;
    marker.scale.z = 2 * obstacles[obs_id].radius;
    marker.color.a = 1;
    msg.markers.emplace_back(marker);
  }

  pub_pose->publish(msg);
}

void ApfAgent::publish_distance_metrics() {
  if (agent_id != 0 || !pub_distance_metrics) {
    return;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  double min_agent_obstacle_dist = nan;
  double min_agent_agent_dist = nan;
  double min_agent_obstacle_agent_id = -1.0;
  double min_agent_obstacle_obs_id = -1.0;
  double min_agent_agent_i = -1.0;
  double min_agent_agent_j = -1.0;

  std_msgs::msg::Float64MultiArray msg;
  msg.data.resize(6 + number_of_agents * number_of_obstacles);

  size_t data_id = 6;
  for (size_t id = 0; id < number_of_agents; id++) {
    for (size_t obs_id = 0; obs_id < number_of_obstacles; obs_id++) {
      Vector3d diff = agent_positions[id] - obstacles[obs_id].position;
      diff.z() = 0.0;
      double surface_dist = diff.norm() - radius - obstacles[obs_id].radius;
      msg.data[data_id++] = surface_dist;

      if (std::isnan(min_agent_obstacle_dist) ||
          surface_dist < min_agent_obstacle_dist) {
        min_agent_obstacle_dist = surface_dist;
        min_agent_obstacle_agent_id = static_cast<double>(id);
        min_agent_obstacle_obs_id = static_cast<double>(obs_id);
      }
    }
  }

  for (size_t i = 0; i < number_of_agents; i++) {
    for (size_t j = i + 1; j < number_of_agents; j++) {
      Vector3d diff = agent_positions[i] - agent_positions[j];
      diff.z() = 0.0;
      double surface_dist = diff.norm() - 2.0 * radius;

      if (std::isnan(min_agent_agent_dist) ||
          surface_dist < min_agent_agent_dist) {
        min_agent_agent_dist = surface_dist;
        min_agent_agent_i = static_cast<double>(i);
        min_agent_agent_j = static_cast<double>(j);
      }
    }
  }

  msg.data[0] = min_agent_obstacle_dist;
  msg.data[1] = min_agent_agent_dist;
  msg.data[2] = min_agent_obstacle_agent_id;
  msg.data[3] = min_agent_obstacle_obs_id;
  msg.data[4] = min_agent_agent_i;
  msg.data[5] = min_agent_agent_j;

  pub_distance_metrics->publish(msg);
}

void ApfAgent::publish_motion_metrics() {
  if (!pub_motion_metrics) {
    return;
  }

  Vector3d velocity = state.velocity;
  velocity.z() = 0.0;
  Vector3d acceleration = last_control_input;
  acceleration.z() = 0.0;

  std_msgs::msg::Float64MultiArray msg;
  msg.layout.dim.resize(1);
  msg.layout.dim[0].label = "motion_metrics_agent_v1";
  msg.layout.dim[0].size = 3;
  msg.layout.dim[0].stride = 3;
  msg.data.resize(3);
  msg.data[0] = static_cast<double>(agent_id);
  msg.data[1] = velocity.norm();
  msg.data[2] = acceleration.norm();
  pub_motion_metrics->publish(msg);
}
} // namespace apf
