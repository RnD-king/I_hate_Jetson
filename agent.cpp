#include "agent.h"

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
  std::cout << "[ApfAgent] Agent" << agent_id << " is ready." << std::endl;
}

void ApfAgent::timer_tf_callback() {
  listen_tf();
  update_state();
  broadcast_tf();
}

void ApfAgent::timer_pub_callback() {
  publish_marker_pose();
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
      double elapsed_time = (this->get_clock()->now() - debug_start_time).seconds();
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
  const double eps = 1e-6;

  const double k_goal = 2.0;
  const double k_obs = 8.0;
  const double k_damp = 2.2;
  const double goal_switch_dist = 1.5;
  const double obs_safety_margin = 0.35;
  const double obs_lookahead_time = 3.0;
  const double min_prediction_speed = 0.5;
  const double k_agent = 0.08;
  const double k_agent_tangent = 2.5;
  const double k_agent_brake = 5.0;
  const double agent_safety_margin = 0.2;
  const double agent_lookahead_time = 2.0;
  const double agent_influence_dist = 1.2;
  const double k_orbit = 1.8;
  const double orbit_min_radius = 0.0;
  const double orbit_max_radius = 4.0;
  const double orbit_goal_cutoff_dist = 1.2;

  // Attraction force
  Vector3d goal_error = goal - state.position;
  goal_error.z() = 0.0;
  double goal_dist = goal_error.norm();
  Vector3d u_goal = Vector3d::Zero();
  if (goal_dist > eps) {
    if (goal_dist > goal_switch_dist) {
      u_goal = k_goal * goal_switch_dist * goal_error / goal_dist;
    } else {
      u_goal = k_goal * goal_error;
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
    double collision_radius = radius + obstacles[obs_id].radius + obs_safety_margin;

    if (time_to_closest < obs_lookahead_time && lateral_dist < collision_radius) {
      Vector3d tangent(to_obs.y(), -to_obs.x(), 0.0);
      double tangent_norm = tangent.norm();
      if (tangent_norm > eps) {
        tangent = tangent / tangent_norm;
      } else {
        tangent = Vector3d(heading.y(), -heading.x(), 0.0);
      }

      double clearance_weight = (collision_radius - lateral_dist) / collision_radius;
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

    if (time_to_closest < agent_lookahead_time && lateral_dist < collision_radius) {
      Vector3d tangent(to_agent.y(), -to_agent.x(), 0.0);
      double tangent_norm = tangent.norm();
      if (tangent_norm > eps) {
        tangent = tangent / tangent_norm;
      } else {
        tangent = Vector3d(heading.y(), -heading.x(), 0.0);
      }

      double clearance_weight = (collision_radius - lateral_dist) / collision_radius;
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
    double orbit_weight =
        0.35 + 0.65 * (center_radius - orbit_min_radius) /
                   (orbit_max_radius - orbit_min_radius);
    if (orbit_weight > 1.0) {
      orbit_weight = 1.0;
    } else if (orbit_weight < 0.0) {
      orbit_weight = 0.0;
    }

    double goal_weight = (goal_dist - orbit_goal_cutoff_dist) / orbit_goal_cutoff_dist;
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
  Vector3d u_damp = -k_damp * state.velocity;
  u_damp.z() = 0.0;

  // Net force
  u = u_goal + u_obs + u_agent + u_damp;
  u.z() = 0.0;

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
} // namespace apf
