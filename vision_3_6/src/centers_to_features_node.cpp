// vision/src/centers_to_features_node.cpp
//
// 중심점(px) -> feature 계산 -> publish
// (YOLO/이미지/IMU/보정 과정은 전부 제거)
//
// 입력:  /line_centers_px (std_msgs/Float32MultiArray)
//   data = [u0, v0, u1, v1, ...]  (픽셀 좌표, float)
// 출력:  /line_features (std_msgs/Float32MultiArray)
//   data = [ey_norm, epsi_norm, kappa_norm, n_norm, L_norm, res_rms_norm, d_ey_norm, d_epsi_norm, v_prev_norm, w_prev_norm]

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

    // ----------------------------
    // Params (feature computation)
    // ----------------------------
    declare_parameter<double>("v_ref_ratio", 0.80);         // v_ref = v_ref_ratio * H
    declare_parameter<double>("theta_max_deg", 60.0);       // clamp/normalize epsi by this
    declare_parameter<double>("res_rms_max_px", 20.0);      // normalize residual RMS by this
    declare_parameter<double>("d_ey_max_norm", 0.20);       // clamp delta ey (already normalized) by this
    declare_parameter<double>("d_epsi_max_deg", 15.0);      // clamp delta epsi by this (deg)
    declare_parameter<double>("kappa_a_max", 0.00003);      // clamp/normalize quadratic 'a' coefficient by this

    v_ref_ratio_     = get_parameter("v_ref_ratio").as_double();
    theta_max_rad_   = get_parameter("theta_max_deg").as_double() * kDeg2Rad;
    res_rms_max_px_  = get_parameter("res_rms_max_px").as_double();
    d_ey_max_norm_   = get_parameter("d_ey_max_norm").as_double();
    d_epsi_max_rad_  = get_parameter("d_epsi_max_deg").as_double() * kDeg2Rad;
    kappa_a_max_     = get_parameter("kappa_a_max").as_double();

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
  static float ClampF(float v, float lo, float hi)
  {
    return std::max(lo, std::min(hi, v));
  }

  static float SafeDivF(float num, float den, float fallback = 0.0f)
  {
    if (std::fabs(den) < 1e-9f) return fallback;
    return num / den;
  }

  bool FitLine_(const std::vector<cv::Point2f>& pts, cv::Vec4f& line) const
  {
    if (pts.size() < 2) return false;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
    const float vx = line[0];
    const float vy = line[1];
    return (std::isfinite(vx) && std::isfinite(vy) && (std::fabs(vx) + std::fabs(vy) > 1e-6f));
  }

  float LineResidualRms_(const std::vector<cv::Point2f>& pts, const cv::Vec4f& line) const
  {
    const float vx = line[0];
    const float vy = line[1];
    const float x0 = line[2];
    const float y0 = line[3];
    const float dn = std::sqrt(vx * vx + vy * vy);
    if (dn < 1e-6f) return 0.0f;

    double sum_sq = 0.0;
    for (const auto& p : pts)
    {
      // distance from point to line defined by (x0,y0) and direction (vx,vy)
      const float dx = p.x - x0;
      const float dy = p.y - y0;
      const float cross = dx * vy - dy * vx;
      const float d = std::fabs(cross) / dn;
      sum_sq += static_cast<double>(d) * static_cast<double>(d);
    }
    const double mean_sq = sum_sq / std::max<size_t>(1, pts.size());
    return static_cast<float>(std::sqrt(mean_sq));
  }

  bool FitQuadraticUofV_(const std::vector<cv::Point2f>& pts, double& a, double& b, double& c) const
  {
    if (pts.size() < 3) return false;

    // Fit u = a*v^2 + b*v + c  (u=x, v=y)
    const int n = static_cast<int>(pts.size());
    cv::Mat A(n, 3, CV_64F);
    cv::Mat y(n, 1, CV_64F);

    for (int i = 0; i < n; ++i)
    {
      const double v = static_cast<double>(pts[i].y);
      const double u = static_cast<double>(pts[i].x);
      A.at<double>(i, 0) = v * v;
      A.at<double>(i, 1) = v;
      A.at<double>(i, 2) = 1.0;
      y.at<double>(i, 0) = u;
    }

    cv::Mat x;
    const bool ok = cv::solve(A, y, x, cv::DECOMP_SVD);
    if (!ok || x.rows != 3) return false;

    a = x.at<double>(0, 0);
    b = x.at<double>(1, 0);
    c = x.at<double>(2, 0);
    return (std::isfinite(a) && std::isfinite(b) && std::isfinite(c));
  }

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
    const int W = image_width_;
    const int H = image_height_;
    const float u_ref = 0.5f * static_cast<float>(W);
    const float v_ref = static_cast<float>(H) * static_cast<float>(ClampF(static_cast<float>(v_ref_ratio_), 0.0f, 1.0f));

    // Basic visibility stats
    const size_t n = centers_px.size();
    float v_min = std::numeric_limits<float>::infinity();
    float v_max = -std::numeric_limits<float>::infinity();
    for (const auto& p : centers_px)
    {
      v_min = std::min(v_min, p.y);
      v_max = std::max(v_max, p.y);
    }

    const float n_norm = ClampF(static_cast<float>(n) / 5.0f, 0.0f, 1.0f);
    const float L_norm = ClampF(SafeDivF((v_max - v_min), static_cast<float>(H), 0.0f), 0.0f, 1.0f);

    // Line fit for ey/epsi/residual (robust, works with >=2 points)
    float ey_norm = 0.0f;
    float epsi_norm = 0.0f;
    float res_rms_norm = 0.0f;

    cv::Vec4f line;
    const bool has_line = FitLine_(centers_px, line);
    float theta = 0.0f;

    if (has_line)
    {
      const float vx = line[0];
      const float vy = line[1];
      const float x0 = line[2];
      const float y0 = line[3];

      // heading angle relative to +v axis (so vertical line => 0)
      theta = std::atan2(vx, vy);
      const float theta_max = static_cast<float>(std::max(1e-6, theta_max_rad_));
      epsi_norm = ClampF(theta / theta_max, -1.0f, 1.0f);

      // u at v_ref: x = x0 + t*vx, y = y0 + t*vy => t=(v_ref-y0)/vy
      const float t = SafeDivF((v_ref - y0), vy, 0.0f);
      const float u_line = x0 + t * vx;

      // lateral error in image (u direction), normalized
      ey_norm = SafeDivF((u_line - u_ref), (0.5f * static_cast<float>(W)), 0.0f);
      ey_norm = ClampF(ey_norm, -1.0f, 1.0f);

      // residual RMS (px) normalized
      const float rms_px = LineResidualRms_(centers_px, line);
      const float denom = static_cast<float>(std::max(1e-6, res_rms_max_px_));
      res_rms_norm = ClampF(rms_px / denom, 0.0f, 1.0f);
    }
    else
    {
      // If not enough points for a line, keep errors at 0, residual at 0
      ey_norm = 0.0f;
      epsi_norm = 0.0f;
      res_rms_norm = 0.0f;
    }

    // Quadratic fit for curvature (works with >=3 points)
    float kappa_norm = 0.0f;
    if (centers_px.size() >= 3)
    {
      double a = 0.0, b = 0.0, c = 0.0;
      if (FitQuadraticUofV_(centers_px, a, b, c))
      {
        const double denom = std::max(1e-12, kappa_a_max_);
        kappa_norm = static_cast<float>(a / denom);
        kappa_norm = ClampF(kappa_norm, -1.0f, 1.0f);
      }
    }

    // temporal deltas (simple, normalized/clipped)
    float d_ey_norm = 0.0f;
    float d_epsi_norm = 0.0f;
    if (prev_valid_)
    {
      d_ey_norm = ey_norm - prev_ey_norm_;
      const float d_ey_max = static_cast<float>(std::max(1e-6, d_ey_max_norm_));
      d_ey_norm = ClampF(d_ey_norm / d_ey_max, -1.0f, 1.0f);

      const float d_theta = theta - prev_theta_rad_;
      const float d_theta_max = static_cast<float>(std::max(1e-6, d_epsi_max_rad_));
      d_epsi_norm = ClampF(d_theta / d_theta_max, -1.0f, 1.0f);
    }

    // previous action (placeholder; kept as internal state)
    const float v_prev_norm = ClampF(v_prev_norm_, -1.0f, 1.0f);
    const float w_prev_norm = ClampF(w_prev_norm_, -1.0f, 1.0f);

    // update prev
    prev_valid_ = true;
    prev_ey_norm_ = ey_norm;
    prev_theta_rad_ = theta;

    // publish (line_perception_node와 동일 포맷)
    std_msgs::msg::Float32MultiArray out;
    out.data.reserve(10);

    out.data.push_back(ey_norm);
    out.data.push_back(epsi_norm);
    out.data.push_back(kappa_norm);
    out.data.push_back(n_norm);
    out.data.push_back(L_norm);
    out.data.push_back(res_rms_norm);
    out.data.push_back(d_ey_norm);
    out.data.push_back(d_epsi_norm);
    out.data.push_back(v_prev_norm);
    out.data.push_back(w_prev_norm);

    features_pub_->publish(out);
    ++pub_count_;

    // 로그 (원하면 삭제/축소)
    RCLCPP_INFO(get_logger(),
      "[PUB] /line_features | ey=%.4f epsi=%.4f kappa=%.4f n=%.2f L=%.2f res=%.3f d_ey=%.3f d_epsi=%.3f v_prev=%.3f w_prev=%.3f (N=%zu)",
      ey_norm, epsi_norm, kappa_norm,
      n_norm, L_norm, res_rms_norm,
      d_ey_norm, d_epsi_norm,
      v_prev_norm, w_prev_norm,
      centers_px.size());
  }

  void PublishEmpty_()
  {
    std_msgs::msg::Float32MultiArray out;
    out.data = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
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

  double v_ref_ratio_{0.80};
  double theta_max_rad_{60.0 * kDeg2Rad};
  double res_rms_max_px_{20.0};
  double d_ey_max_norm_{0.20};
  double d_epsi_max_rad_{15.0 * kDeg2Rad};
  double kappa_a_max_{0.00003};

  bool prev_valid_{false};
  float prev_ey_norm_{0.0f};
  float prev_theta_rad_{0.0f};

  float v_prev_norm_{0.0f};
  float w_prev_norm_{0.0f};

  size_t recv_count_{0};
  size_t pub_count_{0};

  static constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
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