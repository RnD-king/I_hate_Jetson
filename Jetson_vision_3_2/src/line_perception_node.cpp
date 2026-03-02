// vision/src/line_perception_node.cpp
//
// 단일 ROS 2 노드에서 파이프라인을 조립
// 이미지 -> YOLO(TRT) -> bbox 중심점 -> IMU(roll/pitch) 좌표 보정 -> feature 계산 -> publish
//
// 설계 의도(중요):
// - IMU는 시간 동기화하지 않고 “가장 최신값(latest)”만 사용한다.
// - 출력 메시지는 임시로 std_msgs/Float32MultiArray를 사용한다.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdio>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

    // YOLO 엔진 경로 (사용자가 수정한다고 했으므로 그대로 둠)
    const std::string engine_path = "/home/rnd/rnd/yolo/engine/best_8s_16.engine";

    line_class_id_ = 0;
    conf_thres_    = 0.20f;

    // 카메라 내참수
    const double fx = 600.0;
    const double fy = 600.0;
    const double cx = 320.0;
    const double cy = 240.0;

    // IMU 제한값
    imu_abs_limit_rad_ = Deg2Rad(45.0);

    if (engine_path.empty())
    {
      throw std::runtime_error("YOLO TensorRT engine 경로(engine_path)가 비어 있습니다.");
    }

    yolo_ = std::make_unique<YoloTrtEngine>(engine_path);
    point_extractor_ = std::make_unique<LinePointExtractor>(line_class_id_, conf_thres_);

    CoordinateRectifier::Intrinsics K{};
    K.fx = fx; K.fy = fy; K.cx = cx; K.cy = cy;
    rectifier_ = std::make_unique<CoordinateRectifier>(K);

    feature_extractor_ = std::make_unique<FeatureExtractor>();

    // ----------------------------
    // ROS Pub/Sub
    // ----------------------------
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    image_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    image_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, image_qos,
      std::bind(&LinePerceptionNode::OnImage, this, std::placeholders::_1));

    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    imu_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    imu_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    imu_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      imu_topic, imu_qos,
      std::bind(&LinePerceptionNode::OnImu, this, std::placeholders::_1));

    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      features_topic, rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "vision node started.");
    RCLCPP_INFO(get_logger(), "  image_topic    : %s", image_topic.c_str());
    RCLCPP_INFO(get_logger(), "  imu_topic      : %s", imu_topic.c_str());
    RCLCPP_INFO(get_logger(), "  features_topic : %s", features_topic.c_str());
    RCLCPP_INFO(get_logger(), "  engine_path    : %s", engine_path.c_str());

    if (show_debug_view_)
    {
      cv::namedWindow(kDebugWindowName, cv::WINDOW_NORMAL);
      RCLCPP_INFO(get_logger(), "[SETUP] debug window created: %s", kDebugWindowName);
    }

    // ---- heartbeat(1Hz): 로그는 여기서만 출력 ----
    hb_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&LinePerceptionNode::OnHeartbeat, this));

    // ---- PERF(1초 갱신) 초기화: "오버레이만" ----
    last_report_time_ = this->get_clock()->now();
    perf_frame_count_ = 0;
    perf_total_time_sec_ = 0.0;
    perf_text_ = "PING: -- ms | FPS: --";

    // 요약 통계 초기화
    ResetIntervalStats();
  }

  ~LinePerceptionNode() override
  {
    if (show_debug_view_)
    {
      try { cv::destroyWindow(kDebugWindowName); }
      catch (...) {}
    }
  }

private:
  static constexpr const char* kDebugWindowName = "LinePerception Debug";
  static double Deg2Rad(const double deg) { return deg * M_PI / 180.0; }

  // ----------------------------
  // 1Hz heartbeat에서만 로그 출력
  // ----------------------------
  void OnHeartbeat()
  {
    // interval stats는 콜백에서 업데이트되므로, 락으로 스냅샷
    IntervalStats snap;
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      snap = interval_stats_;
      ResetIntervalStatsNoLock();
    }

    // IMU 최신값 스냅샷(참고 출력용)
    double roll = 0.0, pitch = 0.0;
    bool imu_ready_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      roll  = last_roll_rad_;
      pitch = last_pitch_rad_;
      imu_ready_snapshot = imu_ready_;
    }

    // 1초 동안의 요약 로그
    RCLCPP_INFO(
      get_logger(),
      "[HB] frames=%zu imu=%zu pub=%zu | img_ok=%zu img_fail=%zu | yolo_ok=%zu yolo_fail=%zu | cvb_fail=%zu | imu_ready=%d roll=%.5f pitch=%.5f | det_last=%zu centers_last=%zu",
      snap.frames,
      snap.imu_msgs,
      snap.published,
      snap.img_ok,
      snap.img_fail,
      snap.yolo_ok,
      snap.yolo_fail,
      snap.cv_bridge_fail,
      imu_ready_snapshot ? 1 : 0,
      roll, pitch,
      snap.last_det_count,
      snap.last_centers_count
    );

    RCLCPP_INFO(
      get_logger(),
      "[HB-TIME] yolo_last=%.3fms pts_last=%.3fms rect_last=%.3fms feat_last=%.3fms total_last=%.3fms | total_avg=%.3fms (n=%zu)",
      snap.last_yolo_ms,
      snap.last_pts_ms,
      snap.last_rect_ms,
      snap.last_feat_ms,
      snap.last_total_ms,
      (snap.total_count > 0) ? (snap.total_sum_ms / static_cast<double>(snap.total_count)) : 0.0,
      snap.total_count
    );

    // 최근 에러 이유가 있으면 1초에 1번만 출력
    if (!snap.last_error.empty())
    {
      RCLCPP_INFO(get_logger(), "[HB-ERR] last_error=%s", snap.last_error.c_str());
    }
  }

  // ----------------------------
  // IMU callback: 로그 출력 금지(1Hz로만)
  // ----------------------------
  void OnImu(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.imu_msgs += 1;
    }

    std::lock_guard<std::mutex> lock(imu_mutex_);

    double r = static_cast<double>(msg->vector.x);
    double p = static_cast<double>(msg->vector.y);

    r = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, r));
    p = std::max(-imu_abs_limit_rad_, std::min(imu_abs_limit_rad_, p));

    last_roll_rad_  = r;
    last_pitch_rad_ = p;
    imu_ready_ = true;
  }

  void DrawDetections(cv::Mat& vis, const std::vector<Detection>& dets)
  {
    for (const auto& det : dets)
    {
      if (det.class_id != line_class_id_) continue;
      if (det.confidence < conf_thres_) continue;

      const cv::Rect box = det.box;
      cv::Rect clipped = box & cv::Rect(0, 0, vis.cols, vis.rows);
      if (clipped.width <= 0 || clipped.height <= 0) continue;

      const int cx = clipped.x + clipped.width / 2;
      const int cy = clipped.y + clipped.height / 2;

      cv::rectangle(vis, clipped, cv::Scalar(0, 255, 0), 2);
      cv::circle(vis, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);

      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", det.confidence);

      int tx = clipped.x;
      int ty = clipped.y - 6;
      if (ty < 12) ty = clipped.y + 16;

      cv::putText(vis, buf, cv::Point(tx, ty),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    }
  }

  // PERF: 1초에 1번만 문자열 갱신 (로그로는 절대 안 찍음)
  void UpdatePerfOverlayOncePerSecond(double frame_time_sec)
  {
    const rclcpp::Time now = this->get_clock()->now();
    const rclcpp::Duration elapsed = now - last_report_time_;

    perf_frame_count_ += 1;
    perf_total_time_sec_ += frame_time_sec;

    if (elapsed.seconds() >= 1.0)
    {
      const int denom = std::max(1, perf_frame_count_);
      const double avg_ping_ms = (perf_total_time_sec_ / static_cast<double>(denom)) * 1000.0;
      const double fps = static_cast<double>(perf_frame_count_) / elapsed.seconds();

      last_report_time_ = now;
      perf_frame_count_ = 0;
      perf_total_time_sec_ = 0.0;

      char buf[128];
      std::snprintf(buf, sizeof(buf), "PING: %.2fms | FPS: %.2f", avg_ping_ms, fps);
      perf_text_ = buf;
    }
  }

  // ----------------------------
  // Image callback: 로그 출력 금지(1Hz로만)
  // ----------------------------
  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.frames += 1;
    }

    last_img_stamp_sec_ =
      static_cast<double>(msg->header.stamp.sec) +
      static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // 1) ROS Image -> cv::Mat(BGR8)
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    }
    catch (const cv_bridge::Exception& e)
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.cv_bridge_fail += 1;
      interval_stats_.img_fail += 1;
      interval_stats_.last_error = std::string("cv_bridge: ") + e.what();
      return;
    }

    const cv::Mat& bgr = cv_ptr->image;
    if (bgr.empty())
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.img_fail += 1;
      interval_stats_.last_error = "empty image";
      return;
    }

    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.img_ok += 1;
    }

    // 2) IMU 최신값 스냅샷
    double roll = 0.0, pitch = 0.0;
    bool imu_ready_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      roll  = last_roll_rad_;
      pitch = last_pitch_rad_;
      imu_ready_snapshot = imu_ready_;
    }

    // 3) YOLO 추론
    std::vector<Detection> dets;

    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = yolo_->Infer(bgr, dets);
    const auto t2 = std::chrono::steady_clock::now();

    const auto dt_yolo_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (!ok)
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.yolo_fail += 1;
      interval_stats_.last_error = "yolo Infer failed (ok=false)";
      // 디버그 뷰는 그대로 유지하되, 실패 시에는 그리지 않음(원 코드의 흐름 유지)
      return;
    }

    // 4) Detection -> bbox 중심점
    const auto t3 = std::chrono::steady_clock::now();
    std::vector<cv::Point2f> centers_px = point_extractor_->ExtractCenters(dets);
    const auto t4 = std::chrono::steady_clock::now();
    const auto dt_pts_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // 5) 좌표 보정
    std::vector<cv::Point2f> centers_rect_px = centers_px;
    const auto t5 = std::chrono::steady_clock::now();
    if (imu_ready_snapshot)
    {
      centers_rect_px = rectifier_->RectifyPixelPoints(centers_px, roll, pitch);
    }
    const auto t6 = std::chrono::steady_clock::now();
    const auto dt_rec_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    // 6) Feature 계산
    const auto t7 = std::chrono::steady_clock::now();
    FeatureExtractor::Features feats = feature_extractor_->Compute(centers_rect_px, bgr.size());
    const auto t8 = std::chrono::steady_clock::now();
    const auto dt_feat_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();

    // 7) Publish
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
    // out.data.push_back(static_cast<float>(feats.mean_conf));

    features_pub_->publish(out);

    const auto dt_all_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t8 - t0).count();

    // ===== PERF overlay 업데이트(1초에 1번만 문자열 갱신) =====
    UpdatePerfOverlayOncePerSecond(static_cast<double>(dt_all_us) * 1e-6);

    // ---- interval stats 업데이트 (락 잡는 구간 최소화) ----
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      interval_stats_.yolo_ok += 1;
      interval_stats_.published += 1;

      interval_stats_.last_det_count = dets.size();
      interval_stats_.last_centers_count = centers_px.size();

      interval_stats_.last_yolo_ms  = static_cast<double>(dt_yolo_us) / 1000.0;
      interval_stats_.last_pts_ms   = static_cast<double>(dt_pts_us)  / 1000.0;
      interval_stats_.last_rect_ms  = static_cast<double>(dt_rec_us)  / 1000.0;
      interval_stats_.last_feat_ms  = static_cast<double>(dt_feat_us) / 1000.0;
      interval_stats_.last_total_ms = static_cast<double>(dt_all_us)  / 1000.0;

      interval_stats_.total_sum_ms += interval_stats_.last_total_ms;
      interval_stats_.total_count  += 1;
    }

    // ---- 시각화 ----
    if (show_debug_view_)
    {
      cv::Mat vis = bgr.clone();
      DrawDetections(vis, dets);

      // det 요약(매 프레임 오버레이는 OK)
      {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "det=%zu th=%.2f", dets.size(), conf_thres_);
        cv::putText(vis, buf, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
      }

      // perf_text_ (1초에 1번만 갱신되는 문자열)
      cv::putText(vis, perf_text_, cv::Point(10, 60),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

      cv::imshow(kDebugWindowName, vis);
      cv::waitKey(1);
    }
  }

private:
  // ----------------------------
  // interval stats 구조체
  // ----------------------------
  struct IntervalStats
  {
    // counts (1초 단위)
    size_t frames{0};
    size_t imu_msgs{0};
    size_t published{0};

    size_t img_ok{0};
    size_t img_fail{0};
    size_t cv_bridge_fail{0};

    size_t yolo_ok{0};
    size_t yolo_fail{0};

    // last snapshots
    size_t last_det_count{0};
    size_t last_centers_count{0};

    double last_yolo_ms{0.0};
    double last_pts_ms{0.0};
    double last_rect_ms{0.0};
    double last_feat_ms{0.0};
    double last_total_ms{0.0};

    // avg total over interval
    double total_sum_ms{0.0};
    size_t total_count{0};

    // last error string within the interval
    std::string last_error{};
  };

  void ResetIntervalStats()
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ResetIntervalStatsNoLock();
  }

  void ResetIntervalStatsNoLock()
  {
    interval_stats_ = IntervalStats{};
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

  // 디버그 카운터(누적)
  double last_img_stamp_sec_{0.0};

  // ---- PERF 오버레이(1초 갱신) ----
  rclcpp::Time last_report_time_{0, 0, RCL_ROS_TIME};
  int perf_frame_count_{0};
  double perf_total_time_sec_{0.0};
  std::string perf_text_;

  // ---- 시각화/필터 설정(멤버) ----
  int line_class_id_{0};
  float conf_thres_{0.20f};
  bool show_debug_view_{true};

  // ---- 1초 요약 통계 ----
  std::mutex stats_mutex_;
  IntervalStats interval_stats_;
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
