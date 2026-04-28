// 현재는 필요 없. line_perception에서 담당 중

// vision/src/centers_to_features_node.cpp
//
// 중심점(px) -> g1_vision policy 8D feature 계산 -> publish
//
// 입력:
//   /line_centers_px (std_msgs/Float32MultiArray)
//     data = [u0, v0, u1, v1, ...]
//   /g1_vision/cmd_vel (geometry_msgs/Twist, optional)
//     previous policy command feedback for vx_prev/wz_prev
//
// 출력:
//   /g1_vision/features (std_msgs/Float32MultiArray)
//     data = [u_err_near, u_err_lookahead, u_err_ctrl, slope,
//             n_visible, in_recovery, vx_prev, wz_prev]

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <opencv2/core.hpp>

#include "vision/feature_extractor.hpp"

namespace vision {

class CentersToFeaturesNode final : public rclcpp::Node {
public:
  CentersToFeaturesNode() : rclcpp::Node("centers_to_features_node") {
    declare_parameter<std::string>("centers_topic", "/line_centers_px");
    declare_parameter<std::string>("features_topic", "/g1_vision/features");
    declare_parameter<std::string>("prev_cmd_topic", "/g1_vision/cmd_vel");
    declare_parameter<int>("image_width", 640);
    declare_parameter<int>("image_height", 480);
    declare_parameter<int>("max_centers", 8);
    declare_parameter<double>("image_center_u", -1.0);
    declare_parameter<double>("lookahead_delta_v_px", 190.0);
    declare_parameter<double>("lookahead_alpha_normal", 0.70);
    declare_parameter<double>("lookahead_alpha_recovery", 0.20);
    declare_parameter<double>("recover_enter_nvis", 2.0);
    declare_parameter<double>("recover_exit_nvis", 3.0);
    declare_parameter<double>("recover_enter_u", 0.70);
    declare_parameter<double>("recover_exit_u", 0.35);
    declare_parameter<double>("vx_prev_min", 0.0);
    declare_parameter<double>("vx_prev_max", 1.2);
    declare_parameter<double>("wz_prev_min", -1.9);
    declare_parameter<double>("wz_prev_max", 1.9);

    centers_topic_ = get_parameter("centers_topic").as_string();
    features_topic_ = get_parameter("features_topic").as_string();
    prev_cmd_topic_ = get_parameter("prev_cmd_topic").as_string();
    image_width_ = static_cast<int>(get_parameter("image_width").as_int());
    image_height_ = static_cast<int>(get_parameter("image_height").as_int());
    vx_prev_min_ = get_parameter("vx_prev_min").as_double();
    vx_prev_max_ = get_parameter("vx_prev_max").as_double();
    wz_prev_min_ = get_parameter("wz_prev_min").as_double();
    wz_prev_max_ = get_parameter("wz_prev_max").as_double();

    feature_cfg_.max_centers =
        static_cast<int>(get_parameter("max_centers").as_int());
    feature_cfg_.image_center_u = get_parameter("image_center_u").as_double();
    feature_cfg_.lookahead_delta_v_px =
        get_parameter("lookahead_delta_v_px").as_double();
    feature_cfg_.lookahead_alpha_normal =
        get_parameter("lookahead_alpha_normal").as_double();
    feature_cfg_.lookahead_alpha_recovery =
        get_parameter("lookahead_alpha_recovery").as_double();
    feature_cfg_.recover_enter_nvis =
        get_parameter("recover_enter_nvis").as_double();
    feature_cfg_.recover_exit_nvis =
        get_parameter("recover_exit_nvis").as_double();
    feature_cfg_.recover_enter_u = get_parameter("recover_enter_u").as_double();
    feature_cfg_.recover_exit_u = get_parameter("recover_exit_u").as_double();

    feature_extractor_ = std::make_unique<FeatureExtractor>();

    auto centers_qos = rclcpp::QoS(rclcpp::KeepLast(5));
    centers_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    centers_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    centers_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        centers_topic_, centers_qos,
        std::bind(&CentersToFeaturesNode::OnCenters, this,
                  std::placeholders::_1));

    prev_cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        prev_cmd_topic_, rclcpp::QoS(5),
        std::bind(&CentersToFeaturesNode::OnPrevCmd, this,
                  std::placeholders::_1));

    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        features_topic_, rclcpp::QoS(10));

    hb_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {
      RCLCPP_INFO(get_logger(),
                  "[HB] centers_recv=%zu features_pub=%zu prev_cmd_recv=%zu "
                  "recovery=%d",
                  recv_count_, pub_count_, prev_cmd_count_,
                  in_recovery_ ? 1 : 0);
    });

    RCLCPP_INFO(get_logger(), "centers_to_features_node started.");
    RCLCPP_INFO(get_logger(), "  centers_topic  : %s", centers_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  features_topic : %s", features_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  prev_cmd_topic : %s", prev_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  image_size     : %d x %d", image_width_,
                image_height_);
  }

private:
  static double Clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
  }

  void OnPrevCmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
    ++prev_cmd_count_;
    vx_prev_ =
        Clamp(static_cast<double>(msg->linear.x), vx_prev_min_, vx_prev_max_);
    wz_prev_ =
        Clamp(static_cast<double>(msg->angular.z), wz_prev_min_, wz_prev_max_);
  }

  void OnCenters(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    ++recv_count_;

    const auto &data = msg->data;
    if ((data.size() % 2) != 0) {
      RCLCPP_WARN(get_logger(),
                  "[IN] centers array size must be even, got=%zu -> drop",
                  data.size());
      return;
    }

    std::vector<cv::Point2f> centers_px;
    centers_px.reserve(data.size() / 2);
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
      centers_px.emplace_back(data[i], data[i + 1]);
    }

    const auto feats = feature_extractor_->Compute(
        centers_px, cv::Size(image_width_, image_height_), in_recovery_,
        vx_prev_, wz_prev_, feature_cfg_);
    in_recovery_ = feats.in_recovery > 0.5;

    PublishFeatures(feats);
  }

  void PublishFeatures(const FeatureExtractor::Features &feats) {
    std_msgs::msg::Float32MultiArray out;
    out.data.reserve(8);
    out.data.push_back(static_cast<float>(feats.u_err_near));
    out.data.push_back(static_cast<float>(feats.u_err_lookahead));
    out.data.push_back(static_cast<float>(feats.u_err_ctrl));
    out.data.push_back(static_cast<float>(feats.slope));
    out.data.push_back(static_cast<float>(feats.n_visible));
    out.data.push_back(static_cast<float>(feats.in_recovery));
    out.data.push_back(static_cast<float>(feats.vx_prev));
    out.data.push_back(static_cast<float>(feats.wz_prev));

    features_pub_->publish(out);
    ++pub_count_;

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "[PUB] %s | u=%.3f u_la=%.3f u_ctrl=%.3f slope=%.3f "
                         "n=%.0f rec=%.0f vx_prev=%.3f wz_prev=%.3f",
                         features_topic_.c_str(), feats.u_err_near,
                         feats.u_err_lookahead, feats.u_err_ctrl, feats.slope,
                         feats.n_visible, feats.in_recovery, feats.vx_prev,
                         feats.wz_prev);
  }

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
      centers_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr prev_cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr features_pub_;
  rclcpp::TimerBase::SharedPtr hb_timer_;

  std::unique_ptr<FeatureExtractor> feature_extractor_;
  FeatureExtractor::Config feature_cfg_;

  std::string centers_topic_;
  std::string features_topic_;
  std::string prev_cmd_topic_;
  int image_width_{640};
  int image_height_{480};

  bool in_recovery_{false};
  double vx_prev_{0.0};
  double wz_prev_{0.0};
  double vx_prev_min_{0.0};
  double vx_prev_max_{1.2};
  double wz_prev_min_{-1.9};
  double wz_prev_max_{1.9};

  size_t recv_count_{0};
  size_t pub_count_{0};
  size_t prev_cmd_count_{0};
};

} // namespace vision

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<vision::CentersToFeaturesNode>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("vision"), "fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
