// vision/src/line_perception_node.cpp
//
// /camera/color/image_raw
// -> YOLO TensorRT 추론
// -> bbox 중심점 추출
// -> IMU roll/pitch 보정
// -> 8D feature 생성
// -> /line_centers_px publish
// -> /g1_vision/features publish

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vision/coordinate_rectifier.hpp"
#include "vision/feature_extractor.hpp"
#include "vision/line_point_extractor.hpp"
#include "vision/yolo_trt_engine.hpp"

namespace vision {

class LinePerceptionNode final : public rclcpp::Node {
public:
  LinePerceptionNode() : rclcpp::Node("line_perception_node") {
    declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    declare_parameter<std::string>("imu_topic", "/camera/imu_tilt");
    declare_parameter<std::string>("centers_topic", "/line_centers_px");
    declare_parameter<std::string>("features_topic", "/g1_vision/features");
    declare_parameter<std::string>("prev_cmd_topic", "/g1_vision/cmd_vel");
    declare_parameter<std::string>("engine_path",
                                   "/home/noh/my_cv/best_fp32.engine");
    declare_parameter<int>("line_class_id", 0);
    declare_parameter<double>("conf_thres", 0.80);
    declare_parameter<double>("fx", 600.0);
    declare_parameter<double>("fy", 600.0);
    declare_parameter<double>("cx", 320.0);
    declare_parameter<double>("cy", 240.0);
    declare_parameter<bool>("use_imu_rectification", true);
    declare_parameter<bool>("assume_zero_imu", false);
    declare_parameter<double>("imu_abs_limit_deg", 45.0);
    declare_parameter<bool>("show_debug_view", true);
    declare_parameter<double>("inference_hz", 5.0);
    declare_parameter<int>("max_centers", 8);
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

    image_topic_ = get_parameter("image_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    centers_topic_ = get_parameter("centers_topic").as_string();
    features_topic_ = get_parameter("features_topic").as_string();
    prev_cmd_topic_ = get_parameter("prev_cmd_topic").as_string();
    engine_path_ = get_parameter("engine_path").as_string();
    line_class_id_ = static_cast<int>(get_parameter("line_class_id").as_int());
    conf_thres_ = static_cast<float>(get_parameter("conf_thres").as_double());
    use_imu_rectification_ = get_parameter("use_imu_rectification").as_bool();
    assume_zero_imu_ = get_parameter("assume_zero_imu").as_bool();
    imu_abs_limit_rad_ =
        Deg2Rad(get_parameter("imu_abs_limit_deg").as_double());
    show_debug_view_ = get_parameter("show_debug_view").as_bool();
    inference_hz_ = get_parameter("inference_hz").as_double();
    vx_prev_min_ = get_parameter("vx_prev_min").as_double();
    vx_prev_max_ = get_parameter("vx_prev_max").as_double();
    wz_prev_min_ = get_parameter("wz_prev_min").as_double();
    wz_prev_max_ = get_parameter("wz_prev_max").as_double();

    const double fx = get_parameter("fx").as_double();
    const double fy = get_parameter("fy").as_double();
    const double cx = get_parameter("cx").as_double();
    const double cy = get_parameter("cy").as_double();

    feature_cfg_.max_centers =
        static_cast<int>(get_parameter("max_centers").as_int());
    feature_cfg_.image_center_u = cx;
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

    if (engine_path_.empty()) {
      throw std::runtime_error(
          "YOLO TensorRT engine 경로(engine_path)가 비어 있습니다.");
    }

    yolo_ = std::make_unique<YoloTrtEngine>(engine_path_);
    point_extractor_ =
        std::make_unique<LinePointExtractor>(line_class_id_, conf_thres_);

    CoordinateRectifier::Intrinsics K{};
    K.fx = fx;
    K.fy = fy;
    K.cx = cx;
    K.cy = cy;
    rectifier_ = std::make_unique<CoordinateRectifier>(K);

    feature_extractor_ = std::make_unique<FeatureExtractor>();

    // ----------------------------
    // ROS Pub/Sub
    // ----------------------------
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LinePerceptionNode::OnImage, this, std::placeholders::_1));

    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    imu_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    imu_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    imu_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        imu_topic_, imu_qos,
        std::bind(&LinePerceptionNode::OnImu, this, std::placeholders::_1));

    prev_cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        prev_cmd_topic_, rclcpp::QoS(5),
        std::bind(&LinePerceptionNode::OnPrevCmd, this, std::placeholders::_1));

    centers_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        centers_topic_, rclcpp::QoS(10));

    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        features_topic_, rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "vision node started.");
    RCLCPP_INFO(get_logger(), "  image_topic    : %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  imu_topic      : %s", imu_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  centers_topic  : %s", centers_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  features_topic : %s", features_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  prev_cmd_topic : %s", prev_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  engine_path    : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  inference_hz   : %.2f", inference_hz_);
    RCLCPP_INFO(get_logger(), "  imu_rectify    : %s",
                use_imu_rectification_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  zero_imu       : %s",
                assume_zero_imu_ ? "true" : "false");

    RCLCPP_INFO(get_logger(),
                "[SETUP] image subscription created (SensorDataQoS)");
    RCLCPP_INFO(get_logger(), "[SETUP] imu subscription created");
    RCLCPP_INFO(get_logger(), "[SETUP] features publisher created");

    if (show_debug_view_) {
      cv::namedWindow(kDebugWindowName, cv::WINDOW_NORMAL);
      RCLCPP_INFO(get_logger(), "[SETUP] debug window created: %s",
                  kDebugWindowName);
    }

    // ---- heartbeat(1Hz) ----
    hb_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {
      RCLCPP_INFO(get_logger(),
                  "[HB] frames=%zu imu=%zu pub=%zu (last_img_stamp=%.9f)",
                  frames_count_, imu_count_, pub_count_, last_img_stamp_sec_);
    });

    if (inference_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / inference_hz_));
      inference_timer_ =
          create_wall_timer(period, [this]() { ProcessLatestImage(); });
    }

    // ---- PERF(1초 갱신) 초기화: "오버레이만" ----
    last_report_time_ = this->get_clock()->now();
    perf_frame_count_ = 0;
    perf_total_time_sec_ = 0.0;
    perf_text_ = "PING: -- ms | FPS: --";
  }

  ~LinePerceptionNode() override {
    if (show_debug_view_) {
      try {
        cv::destroyWindow(kDebugWindowName);
      } catch (...) {
      }
    }
  }

private:
  static constexpr const char *kDebugWindowName = "LinePerception Debug";
  static double Deg2Rad(const double deg) { return deg * M_PI / 180.0; }
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

  void OnImu(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
    ++imu_count_;

    std::lock_guard<std::mutex> lock(imu_mutex_);

    double r = static_cast<double>(msg->vector.x);
    double p = static_cast<double>(msg->vector.y);

    r = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, r));
    p = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, p));

    last_roll_rad_ = r;
    last_pitch_rad_ = p;

    imu_ready_ = true;

    RCLCPP_DEBUG(get_logger(),
                 "[IMU] recv roll=%.5f pitch=%.5f (clamped) | imu_ready=%d",
                 last_roll_rad_, last_pitch_rad_, imu_ready_ ? 1 : 0);
  }

  void DrawDetections(cv::Mat &vis, const std::vector<Detection> &dets) {
    for (const auto &det : dets) {
      if (det.class_id != line_class_id_)
        continue;
      if (det.confidence < conf_thres_)
        continue;

      const cv::Rect box = det.box;
      cv::Rect clipped = box & cv::Rect(0, 0, vis.cols, vis.rows);
      if (clipped.width <= 0 || clipped.height <= 0)
        continue;

      const int cx = clipped.x + clipped.width / 2;
      const int cy = clipped.y + clipped.height / 2;

      cv::rectangle(vis, clipped, cv::Scalar(0, 255, 0), 2);
      cv::circle(vis, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);

      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", det.confidence);

      int tx = clipped.x;
      int ty = clipped.y - 6;
      if (ty < 12)
        ty = clipped.y + 16;

      cv::putText(vis, buf, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(255, 0, 0), 2);
    }
  }

  // PERF: 1초에 1번만 문자열 갱신 (로그로는 절대 안 찍음)
  void UpdatePerfOverlayOncePerSecond(double frame_time_sec) {
    const rclcpp::Time now = this->get_clock()->now();
    const rclcpp::Duration elapsed = now - last_report_time_;

    perf_frame_count_ += 1;
    perf_total_time_sec_ += frame_time_sec;

    if (elapsed.seconds() >= 1.0) {
      const int denom = std::max(1, perf_frame_count_);
      const double avg_ping_ms =
          (perf_total_time_sec_ / static_cast<double>(denom)) * 1000.0;
      const double fps =
          static_cast<double>(perf_frame_count_) / elapsed.seconds();

      last_report_time_ = now;
      perf_frame_count_ = 0;
      perf_total_time_sec_ = 0.0;

      char buf[128];
      std::snprintf(buf, sizeof(buf), "PING: %.2fms | FPS: %.2f", avg_ping_ms,
                    fps);
      perf_text_ = buf;
    }
  }

  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    ++frames_count_;
    last_img_stamp_sec_ = static_cast<double>(msg->header.stamp.sec) +
                          static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    if (inference_hz_ > 0.0) {
      std::lock_guard<std::mutex> lock(image_mutex_);
      latest_image_ = msg;
      return;
    }

    ProcessImage(msg);
  }

  void ProcessLatestImage() {
    sensor_msgs::msg::Image::SharedPtr msg;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      msg = latest_image_;
      latest_image_.reset();
    }

    if (!msg) {
      return;
    }

    ProcessImage(msg);
  }

  void ProcessImage(const sensor_msgs::msg::Image::SharedPtr &msg) {
    const auto t0 = std::chrono::steady_clock::now();

    RCLCPP_DEBUG(get_logger(), "[STEP %zu] Process image", frames_count_);

    // 1) ROS Image -> cv::Mat(BGR8)
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "[IMG] cv_bridge 예외: %s", e.what());
      return;
    }

    const cv::Mat &bgr = cv_ptr->image;
    if (bgr.empty()) {
      RCLCPP_WARN(get_logger(), "[IMG] 빈 이미지가 들어왔습니다.");
      return;
    }

    // 2) IMU 최신값 스냅샷
    double roll = 0.0, pitch = 0.0;
    bool imu_ready_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      roll = last_roll_rad_;
      pitch = last_pitch_rad_;
      imu_ready_snapshot = imu_ready_;
    }
    RCLCPP_DEBUG(get_logger(),
                 "[IMU] snapshot roll=%.5f pitch=%.5f | imu_ready=%d", roll,
                 pitch, imu_ready_snapshot ? 1 : 0);

    // 3) YOLO 추론
    std::vector<Detection> dets;

    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = yolo_->Infer(bgr, dets);
    const auto t2 = std::chrono::steady_clock::now();

    const auto dt_yolo_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (!ok) {
      RCLCPP_ERROR(get_logger(),
                   "[YOLO] Infer failed -> skip remaining stages");
      return;
    }

    // 4) Detection -> bbox 중심점
    const auto t3 = std::chrono::steady_clock::now();
    std::vector<cv::Point2f> centers_px =
        point_extractor_->ExtractCenters(dets);
    const auto t4 = std::chrono::steady_clock::now();
    const auto dt_pts_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // 5) 좌표 보정
    std::vector<cv::Point2f> centers_rect_px = centers_px;
    const auto t5 = std::chrono::steady_clock::now();
    if (assume_zero_imu_) {
      roll = 0.0;
      pitch = 0.0;
      imu_ready_snapshot = true;
    }

    if (use_imu_rectification_ && imu_ready_snapshot) {
      centers_rect_px = rectifier_->RectifyPixelPoints(centers_px, roll, pitch);
    }
    const auto t6 = std::chrono::steady_clock::now();
    const auto dt_rec_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    // 6) Publish centers + policy-compatible 8D feature
    PublishCenters(centers_rect_px);

    const auto t7 = std::chrono::steady_clock::now();
    const auto feats =
        feature_extractor_->Compute(centers_rect_px, bgr.size(), in_recovery_,
                                    vx_prev_, wz_prev_, feature_cfg_);
    in_recovery_ = feats.in_recovery > 0.5;
    const auto t8 = std::chrono::steady_clock::now();
    const auto dt_feat_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();

    PublishFeatures(feats);

    const auto dt_all_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t8 - t0).count();

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "[TIME] yolo=%.2fms pts=%.2fms rect=%.2fms "
                         "feat=%.2fms total=%.2fms n=%.0f rec=%.0f",
                         static_cast<double>(dt_yolo_us) / 1000.0,
                         static_cast<double>(dt_pts_us) / 1000.0,
                         static_cast<double>(dt_rec_us) / 1000.0,
                         static_cast<double>(dt_feat_us) / 1000.0,
                         static_cast<double>(dt_all_us) / 1000.0,
                         feats.n_visible, feats.in_recovery);

    // ===== PERF overlay 업데이트(1초에 1번만 문자열 갱신) =====
    UpdatePerfOverlayOncePerSecond(static_cast<double>(dt_all_us) * 1e-6);

    // ---- 시각화 ----
    if (show_debug_view_) {
      cv::Mat vis = bgr.clone();
      DrawDetections(vis, dets);

      // det 요약(매 프레임)
      {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "det=%zu th=%.2f", dets.size(),
                      conf_thres_);
        cv::putText(vis, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);
      }

      // perf_text_ (1초에 1번만 갱신되는 문자열)
      cv::putText(vis, perf_text_, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                  0.8, cv::Scalar(0, 255, 0), 2);

      // ===== IMU overlay =====
      {
        char buf[128];

        if (imu_ready_snapshot) {
          std::snprintf(buf, sizeof(buf),
                        "IMU: TRUE | roll=%.3f rad (%.1f deg) | pitch=%.3f rad "
                        "(%.1f deg)",
                        roll, roll * 180.0 / M_PI, pitch, pitch * 180.0 / M_PI);
        } else {
          std::snprintf(buf, sizeof(buf),
                        "IMU: FALSE | roll=NONE | pitch=NONE");
        }

        cv::putText(vis, buf, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
      }

      cv::imshow(kDebugWindowName, vis);
      cv::waitKey(1);
    }
  }

  void PublishCenters(const std::vector<cv::Point2f> &centers_px) {
    std_msgs::msg::Float32MultiArray out;
    out.data.reserve(centers_px.size() * 2);
    for (const auto &p : centers_px) {
      out.data.push_back(p.x);
      out.data.push_back(p.y);
    }
    centers_pub_->publish(out);
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

private:
  // ROS Sub/Pub
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr prev_cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr centers_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr features_pub_;
  rclcpp::TimerBase::SharedPtr hb_timer_;
  rclcpp::TimerBase::SharedPtr inference_timer_;

  // 모듈
  std::unique_ptr<YoloTrtEngine> yolo_;
  std::unique_ptr<LinePointExtractor> point_extractor_;
  std::unique_ptr<CoordinateRectifier> rectifier_;
  std::unique_ptr<FeatureExtractor> feature_extractor_;

  // IMU 최신 상태(latest)
  std::mutex imu_mutex_;
  bool imu_ready_{false};
  bool use_imu_rectification_{false};
  bool assume_zero_imu_{false};
  double last_roll_rad_{0.0};
  double last_pitch_rad_{0.0};
  double imu_abs_limit_rad_{Deg2Rad(45.0)};

  FeatureExtractor::Config feature_cfg_;
  std::mutex image_mutex_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  bool in_recovery_{false};
  double vx_prev_{0.0};
  double wz_prev_{0.0};
  double vx_prev_min_{0.0};
  double vx_prev_max_{1.2};
  double wz_prev_min_{-1.9};
  double wz_prev_max_{1.9};

  // 디버그 카운터
  size_t frames_count_{0};
  size_t imu_count_{0};
  size_t pub_count_{0};
  size_t prev_cmd_count_{0};
  double last_img_stamp_sec_{0.0};

  // ---- PERF 오버레이(1초 갱신) ----
  rclcpp::Time last_report_time_{0, 0, RCL_ROS_TIME};
  int perf_frame_count_{0};
  double perf_total_time_sec_{0.0};
  std::string perf_text_;

  // ---- 시각화/필터 설정(멤버) ----
  int line_class_id_{0};
  float conf_thres_{0.60f};
  bool show_debug_view_{false};
  double inference_hz_{5.0};

  std::string image_topic_;
  std::string imu_topic_;
  std::string centers_topic_;
  std::string features_topic_;
  std::string prev_cmd_topic_;
  std::string engine_path_;
};

} // namespace vision

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<vision::LinePerceptionNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("vision"), "치명적 오류: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
