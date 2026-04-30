#include "agent.h"

#include <cmath>
#include <limits>

using namespace std::chrono_literals;
namespace apf {
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
  agent_goals.resize(number_of_agents);
  for (size_t id = 0; id < number_of_agents; id++) {
    agent_positions[id] = Vector3d(agents_yaml[id]["start"][0].as<double>(),
                                   agents_yaml[id]["start"][1].as<double>(),
                                   agents_yaml[id]["start"][2].as<double>());
    agent_goals[id] = Vector3d(agents_yaml[id]["goal"][0].as<double>(),
                               agents_yaml[id]["goal"][1].as<double>(),
                               agents_yaml[id]["goal"][2].as<double>());
  }
  start = agent_positions[agent_id];
  goal = agent_goals[agent_id];

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

  const double k_goal = 1.0;
  const double k_obs = 0.12;
  const double k_agent = 0.22;
  const double k_damp = 2.2;
  const double goal_switch_dist = 1.5;
  const double obs_influence_dist = 1.2;
  const double agent_influence_dist = 0.5;

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
    Vector3d diff = state.position - obstacles[obs_id].position;
    diff.z() = 0.0;
    double center_dist = diff.norm();
    Vector3d direction = Vector3d(1.0, 0.0, 0.0);
    if (center_dist > eps) {
      direction = diff / center_dist;
    }

    double surface_dist = center_dist - radius - obstacles[obs_id].radius;
    if (surface_dist < obs_influence_dist) {
      double rho = surface_dist;
      if (rho < eps) {
        rho = eps;
      }
      u_obs += k_obs * (1.0 / rho - 1.0 / obs_influence_dist) / (rho * rho) *
               direction;
    }
  }

  Vector3d u_agent = Vector3d::Zero();
  for (size_t id = 0; id < number_of_agents; id++) {
    if (id == agent_id) {
      continue;
    }

    Vector3d diff = state.position - agent_positions[id];
    diff.z() = 0.0;
    double center_dist = diff.norm();
    Vector3d direction = Vector3d(1.0, 0.0, 0.0);
    if (center_dist > eps) {
      direction = diff / center_dist;
    }

    double surface_dist = center_dist - 2.0 * radius;
    if (surface_dist < agent_influence_dist) {
      double rho = surface_dist;
      if (rho < eps) {
        rho = eps;
      }

      u_agent += k_agent * (1.0 / rho - 1.0 / agent_influence_dist) /
                 (rho * rho) * direction;
    }
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
  size_t goal_data_start = 6 + number_of_agents * number_of_obstacles;
  msg.data.resize(goal_data_start + 1 + number_of_agents);

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

  double goal_dist_sum = 0.0;
  for (size_t id = 0; id < number_of_agents; id++) {
    Vector3d goal_error = agent_goals[id] - agent_positions[id];
    goal_error.z() = 0.0;
    double goal_dist = goal_error.norm();
    goal_dist_sum += goal_dist;
    msg.data[goal_data_start + 1 + id] = goal_dist;
  }
  msg.data[goal_data_start] = goal_dist_sum / number_of_agents;

  pub_distance_metrics->publish(msg);
}
} // namespace apf
