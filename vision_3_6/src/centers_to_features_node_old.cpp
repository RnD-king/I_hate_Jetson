// vision/src/centers_to_features_node.cpp
//
// 중심점(px) -> feature 계산 -> publish
// (YOLO/이미지/IMU/보정 과정은 전부 제거)
//
// 입력:  /line_centers_px (std_msgs/Float32MultiArray)
//   data = [u0, v0, u1, v1, ...]  (픽셀 좌표, float)
// 출력:  /line_features (std_msgs/Float32MultiArray)
//   data = [slope_mean, slope_var, x_mean_norm, x_var_norm, y_mean_norm, y_var_norm, count_norm]

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <opencv2/core.hpp>

#include "vision/feature_extractor.hpp"

namespace vision
{

class CentersToFeaturesNode final : public rclcpp::Node
{
public:
  CentersToFeaturesNode()
  : rclcpp::Node("centers_to_features_node")
  {
    // ----------------------------
    // Topics
    // ----------------------------
    const std::string centers_topic  = "/line_centers_px";
    const std::string features_topic = "/line_features";

    // ----------------------------
    // Params (image size for normalization)
    // ----------------------------
    declare_parameter<int>("image_width",  640);
    declare_parameter<int>("image_height", 480);
    image_width_  = get_parameter("image_width").as_int();
    image_height_ = get_parameter("image_height").as_int();

    feature_extractor_ = std::make_unique<FeatureExtractor>();

    // ----------------------------
    // ROS Pub/Sub
    // ----------------------------
    auto in_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    in_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    in_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    centers_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      centers_topic, in_qos,
      std::bind(&CentersToFeaturesNode::OnCenters, this, std::placeholders::_1));

    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      features_topic, rclcpp::QoS(10));

    // heartbeat (원하면 삭제)
    hb_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      [this]() {
        RCLCPP_INFO(
          get_logger(),
          "[HB] recv=%zu pub=%zu (w=%d h=%d)",
          recv_count_, pub_count_, image_width_, image_height_);
      });

    RCLCPP_INFO(get_logger(), "centers_to_features_node started.");
    RCLCPP_INFO(get_logger(), "  centers_topic  : %s", centers_topic.c_str());
    RCLCPP_INFO(get_logger(), "  features_topic : %s", features_topic.c_str());
    RCLCPP_INFO(get_logger(), "  image_size     : %d x %d", image_width_, image_height_);
  }

private:
  void OnCenters(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    ++recv_count_;

    const auto& data = msg->data;

    if (data.empty())
    {
      PublishEmpty_();
      return;
    }

    if ((data.size() % 2) != 0)
    {
      RCLCPP_WARN(get_logger(),
        "[IN] centers array size must be even, got=%zu -> drop",
        data.size());
      return;
    }

    std::vector<cv::Point2f> centers_px;
    centers_px.reserve(data.size() / 2);

    for (size_t i = 0; i + 1 < data.size(); i += 2)
    {
      const float u = data[i + 0];
      const float v = data[i + 1];
      centers_px.emplace_back(u, v);
    }

    // Feature 계산
    const cv::Size img_size(image_width_, image_height_);
    const FeatureExtractor::Features feats =
      feature_extractor_->Compute(centers_px, img_size);

    // publish (line_perception_node와 동일 포맷)
    std_msgs::msg::Float32MultiArray out;
    out.data.reserve(7);

    out.data.push_back(static_cast<float>(feats.slope_mean));
    out.data.push_back(static_cast<float>(feats.slope_var));
    out.data.push_back(static_cast<float>(feats.x_mean_norm));
    out.data.push_back(static_cast<float>(feats.x_var_norm));
    out.data.push_back(static_cast<float>(feats.y_mean_norm));
    out.data.push_back(static_cast<float>(feats.y_var_norm));
    out.data.push_back(static_cast<float>(feats.count_norm));

    features_pub_->publish(out);
    ++pub_count_;

    // 로그 (원하면 삭제/축소)
    RCLCPP_INFO(get_logger(),
      "[PUB] /line_features | slope_mean=%.4f slope_var=%.4f x_mean=%.4f y_mean=%.4f count_norm=%.4f (N=%zu)",
      feats.slope_mean, feats.slope_var,
      feats.x_mean_norm, feats.y_mean_norm,
      feats.count_norm, centers_px.size());
  }

  void PublishEmpty_()
  {
    std_msgs::msg::Float32MultiArray out;
    out.data = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    features_pub_->publish(out);
    ++pub_count_;
  }

private:
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr centers_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr features_pub_;
  rclcpp::TimerBase::SharedPtr hb_timer_;

  std::unique_ptr<FeatureExtractor> feature_extractor_;

  int image_width_{640};
  int image_height_{480};

  size_t recv_count_{0};
  size_t pub_count_{0};
};

}  // namespace vision

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try
  {
    rclcpp::spin(std::make_shared<vision::CentersToFeaturesNode>());
  }
  catch (const std::exception& e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("vision"), "fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}