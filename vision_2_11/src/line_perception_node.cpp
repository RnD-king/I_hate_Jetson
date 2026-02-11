// vision/src/line_perception_node.cpp
//
// 단일 ROS 2 노드에서 파이프라인을 “조립”만 한다:
// 이미지 -> YOLO(TRT) -> bbox 중심점 -> IMU(roll/pitch) 기반 좌표 보정 -> feature 계산 -> publish
//
// 설계 의도(중요):
// - IMU는 시간 동기화하지 않고 “가장 최신값(latest)”만 사용한다.
//   (토픽/버퍼/동기화 복잡도를 늘리지 않고, 모듈을 순수 C++로 유지하기 위함)
// - 출력 메시지는 임시로 std_msgs/Float32MultiArray를 사용한다.
//   (나중에 커스텀 메시지로 교체해도 모듈 인터페이스는 유지 가능)

// src/yolo_trt_engine.cpp
// src/line_point_extractor.cpp
// src/coordinate_rectifier.cpp
// src/feature_extractor.cpp

#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// 프로젝트 모듈(순수 C++ 클래스)
#include "vision/yolo_trt_engine.hpp"
#include "vision/line_point_extractor.hpp"
#include "vision/coordinate_rectifier.hpp"
#include "vision/feature_extractor.hpp"

namespace vision
{

class LinePerceptionNode final : public rclcpp::Node
{
public:
  LinePerceptionNode()
  : rclcpp::Node("line_perception_node")
  {
    // ----------------------------
    const std::string image_topic    = "/camera/color/image_raw";
    const std::string imu_topic      = "/imu/filtered";
    const std::string features_topic = "/line_features";

    // YOLO 엔진 경로(필수)
    const std::string engine_path = "/home/noh/my_cv/best_fp32.engine";

    // 점선 클래스 및 confidence threshold
    const int   line_class_id = 0;
    const float conf_thres    = 0.25f;

    // 카메라 내참수(필요: 보정에서 광선 투영/회전 등을 쓰는 경우)
    const double fx = 600.0;
    const double fy = 600.0;
    const double cx = 320.0;
    const double cy = 240.0;

    // IMU 이상치로 수치 폭주 방지(roll/pitch 절대값 제한)
    // 단, “판단 로직”이 아니라 비정상 센서값에 대한 안전장치다.
    imu_abs_limit_rad_ = Deg2Rad(45.0);

    // ----------------------------
    // 파이프라인 모듈 구성(순수 C++)
    // ----------------------------
    // 엔진 경로는 비어 있으면 의미가 없으므로 즉시 실패(fail-fast) 처리
    if (engine_path.empty())
    {
      throw std::runtime_error("YOLO TensorRT engine 경로(engine_path)가 비어 있습니다.");
    }

    yolo_ = std::make_unique<YoloTrtEngine>(engine_path);
    point_extractor_ = std::make_unique<LinePointExtractor>(line_class_id, conf_thres);

    CoordinateRectifier::Intrinsics K{};
    K.fx = fx; K.fy = fy; K.cx = cx; K.cy = cy;
    rectifier_ = std::make_unique<CoordinateRectifier>(K);

    feature_extractor_ = std::make_unique<FeatureExtractor>();

    // ----------------------------
    // ROS Pub/Sub
    // ----------------------------
    // 중요: 현재 publisher QoS가 RELIABLE + TRANSIENT_LOCAL 로 찍혔음.
    // SensorDataQoS(BEST_EFFORT + VOLATILE)로는 실제 샘플 delivery가 막히는 케이스가 있으므로
    // 1차 디버그는 publisher에 QoS를 맞춘다.
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    image_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    image_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, image_qos,
      std::bind(&LinePerceptionNode::OnImage, this, std::placeholders::_1));

    // IMU도 일단 publisher에 맞추고 싶으면 동일하게 맞추면 됨.
    // (다만 지금은 IMU를 쓰지 않으니 수신 확인 정도만)
    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    imu_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    imu_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    imu_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      imu_topic, imu_qos,
      std::bind(&LinePerceptionNode::OnImu, this, std::placeholders::_1));

    // feature 출력은 유실 방지 목적에서 depth=10 정도면 충분
    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      features_topic, rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "vision node started.");
    RCLCPP_INFO(get_logger(), "  image_topic    : %s", image_topic.c_str());
    RCLCPP_INFO(get_logger(), "  imu_topic      : %s", imu_topic.c_str());
    RCLCPP_INFO(get_logger(), "  features_topic : %s", features_topic.c_str());
    RCLCPP_INFO(get_logger(), "  engine_path    : %s", engine_path.c_str());

    // ---- 추가: 구독/퍼블리셔 생성 완료 로그 ----
    RCLCPP_INFO(get_logger(), "[SETUP] image subscription created (RELIABLE + TRANSIENT_LOCAL, KeepLast=1)");
    RCLCPP_INFO(get_logger(), "[SETUP] imu subscription created");
    RCLCPP_INFO(get_logger(), "[SETUP] features publisher created");

    // ---- 추가: heartbeat(1Hz) ----
    hb_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      [this]() {
        RCLCPP_INFO(
          get_logger(),
          "[HB] frames=%zu imu=%zu pub=%zu (last_img_stamp=%.9f)",
          frames_count_, imu_count_, pub_count_, last_img_stamp_sec_);
      });
  }

private:
  static double Deg2Rad(const double deg) { return deg * M_PI / 180.0; }

  void OnImu(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    ++imu_count_;

    // 전제: msg->vector.x = roll, msg->vector.y = pitch (rad)
    // “단위(deg/rad)”는 여기서 추측으로 변환하지 않는다.
    // 단위가 deg면 IMU 노드에서 rad로 변환해서 publish하는 편이 더 안전하다(단위 혼선 방지).
    std::lock_guard<std::mutex> lock(imu_mutex_);

    double r = static_cast<double>(msg->vector.x);
    double p = static_cast<double>(msg->vector.y);

    // 센서 순간 튐(spike) 등 비정상 값이 들어올 때 보정 수식이 폭주하는 것을 방지
    r = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, r));
    p = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, p));

    last_roll_rad_  = r;
    last_pitch_rad_ = p;

    // 지금은 IMU를 쓰지 않기로 했으니, 의도적으로 false 유지
    imu_ready_ = false;  // 쓸거면 true

    // 매 주기 출력(어지러워도 OK)
    RCLCPP_INFO(
      get_logger(),
      "[IMU] recv roll=%.5f pitch=%.5f (clamped) | imu_ready=%d",
      last_roll_rad_, last_pitch_rad_, imu_ready_ ? 1 : 0);
  }

  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    ++frames_count_;
    last_img_stamp_sec_ =
      static_cast<double>(msg->header.stamp.sec) +
      static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    RCLCPP_INFO(
      get_logger(),
      "[IMG] #%zu recv %ux%u enc=%s step=%u stamp=%.9f",
      frames_count_, msg->width, msg->height, msg->encoding.c_str(), msg->step, last_img_stamp_sec_);

    const auto t0 = std::chrono::steady_clock::now();

    // 1) ROS Image -> cv::Mat(BGR8)
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      // OpenCV 표준 포맷(BGR8) 사용
      // TRT wrapper 내부에서 BGR->RGB/RGBA 변환이 필요하면 내부에서 처리
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    }
    catch (const cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(get_logger(), "[IMG] cv_bridge 예외: %s", e.what());
      return;
    }

    const cv::Mat& bgr = cv_ptr->image;
    if (bgr.empty())
    {
      RCLCPP_WARN(get_logger(), "[IMG] 빈 이미지가 들어왔습니다.");
      return;
    }

    RCLCPP_INFO(get_logger(), "[IMG] cv::Mat ok: %dx%d type=%d", bgr.cols, bgr.rows, bgr.type());

    // 2) IMU 최신값 스냅샷(동기화 X, latest만 사용)
    double roll = 0.0, pitch = 0.0;
    bool imu_ready_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      roll  = last_roll_rad_;
      pitch = last_pitch_rad_;
      imu_ready_snapshot = imu_ready_;
    }
    RCLCPP_INFO(get_logger(), "[IMU] snapshot roll=%.5f pitch=%.5f | imu_ready=%d",
                roll, pitch, imu_ready_snapshot ? 1 : 0);

    // 3) YOLO 추론
    std::vector<Detection> dets;
    RCLCPP_INFO(get_logger(), "[YOLO] infer start");

    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = yolo_->Infer(bgr, dets);
    const auto t2 = std::chrono::steady_clock::now();

    const auto dt_yolo_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    RCLCPP_INFO(get_logger(), "[YOLO] infer done ok=%d det=%zu time=%.2fms",
                ok ? 1 : 0, dets.size(), static_cast<double>(dt_yolo_us) / 1000.0);

    if (!ok)
    {
      RCLCPP_ERROR(get_logger(), "[YOLO] Infer failed -> skip remaining stages");
      return;
    }

    // 4) Detection -> bbox 중심점(픽셀 좌표)
    RCLCPP_INFO(get_logger(), "[PTS] extract centers start");
    const auto t3 = std::chrono::steady_clock::now();
    std::vector<cv::Point2f> centers_px = point_extractor_->ExtractCenters(dets);
    const auto t4 = std::chrono::steady_clock::now();

    const auto dt_pts_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    RCLCPP_INFO(get_logger(), "[PTS] centers=%zu time=%.2fms",
                centers_px.size(), static_cast<double>(dt_pts_us) / 1000.0);

    // 5) IMU 기반 좌표 보정(픽셀 -> 보정된 픽셀)
    // (현재 imu_ready_가 false로 고정되어 있으므로 보정은 항상 스킵될 것)
    std::vector<cv::Point2f> centers_rect_px = centers_px;
    const auto t5 = std::chrono::steady_clock::now();
    if (imu_ready_snapshot)
    {
      RCLCPP_INFO(get_logger(), "[RECT] apply rectifier start");
      centers_rect_px = rectifier_->RectifyPixelPoints(centers_px, roll, pitch);
      RCLCPP_INFO(get_logger(), "[RECT] apply rectifier done");
    }
    else
    {
      RCLCPP_INFO(get_logger(), "[RECT] skipped (imu_ready_=false)");
    }
    const auto t6 = std::chrono::steady_clock::now();
    const auto dt_rec_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
    RCLCPP_INFO(get_logger(), "[RECT] time=%.2fms", static_cast<double>(dt_rec_us) / 1000.0);

    // 6) Feature 계산(정책 입력용 저차원)
    RCLCPP_INFO(get_logger(), "[FEAT] compute start");
    const auto t7 = std::chrono::steady_clock::now();
    FeatureExtractor::Features feats = feature_extractor_->Compute(centers_rect_px, bgr.size());
    const auto t8 = std::chrono::steady_clock::now();
    const auto dt_feat_us = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
    RCLCPP_INFO(get_logger(), "[FEAT] compute done time=%.2fms", static_cast<double>(dt_feat_us) / 1000.0);

    // 7) Feature publish (임시: Float32MultiArray)
    auto out = std_msgs::msg::Float32MultiArray();
    out.layout.dim.clear();
    out.data.clear();
    out.data.reserve(8);

    out.data.push_back(static_cast<float>(feats.slope_mean));
    out.data.push_back(static_cast<float>(feats.slope_var));
    out.data.push_back(static_cast<float>(feats.x_mean_norm));
    out.data.push_back(static_cast<float>(feats.x_var_norm));
    out.data.push_back(static_cast<float>(feats.y_mean_norm));
    out.data.push_back(static_cast<float>(feats.y_var_norm));
    out.data.push_back(static_cast<float>(feats.count_norm));
    out.data.push_back(static_cast<float>(feats.mean_conf));

    features_pub_->publish(out);
    ++pub_count_;

    RCLCPP_INFO(get_logger(),
      "[PUB] /line_features published | slope_mean=%.4f slope_var=%.4f x_mean=%.4f y_mean=%.4f count_norm=%.4f mean_conf=%.4f",
      feats.slope_mean, feats.slope_var, feats.x_mean_norm, feats.y_mean_norm, feats.count_norm, feats.mean_conf);

    const auto dt_all_us  = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t0).count();
    RCLCPP_INFO(get_logger(),
      "[TIME] yolo=%.2fms pts=%.2fms rect=%.2fms feat=%.2fms total=%.2fms",
      static_cast<double>(dt_yolo_us) / 1000.0,
      static_cast<double>(dt_pts_us)  / 1000.0,
      static_cast<double>(dt_rec_us)  / 1000.0,
      static_cast<double>(dt_feat_us) / 1000.0,
      static_cast<double>(dt_all_us)  / 1000.0);
  }

private:
  // ROS Sub/Pub
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr features_pub_;
  rclcpp::TimerBase::SharedPtr hb_timer_;

  // 모듈
  std::unique_ptr<YoloTrtEngine> yolo_;
  std::unique_ptr<LinePointExtractor> point_extractor_;
  std::unique_ptr<CoordinateRectifier> rectifier_;
  std::unique_ptr<FeatureExtractor> feature_extractor_;

  // IMU 최신 상태(latest)
  std::mutex imu_mutex_;
  bool imu_ready_{false};
  double last_roll_rad_{0.0};
  double last_pitch_rad_{0.0};
  double imu_abs_limit_rad_{Deg2Rad(45.0)};

  // 디버그 카운터
  size_t frames_count_{0};
  size_t imu_count_{0};
  size_t pub_count_{0};
  double last_img_stamp_sec_{0.0};
};

}  // namespace vision

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<vision::LinePerceptionNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception& e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("vision"), "치명적 오류: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}