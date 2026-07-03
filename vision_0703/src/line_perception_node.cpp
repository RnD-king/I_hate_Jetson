// vision/src/line_perception_node.cpp
//
// 1. line_perception_node.cpp
//    전체 ROS 노드 시작점. 이미지/IMU를 받고 전체 파이프라인을 호출함.

// 2. yolo_trt_engine.cpp
//    line_perception_node가 YOLO 추론을 요청하면 실행됨.
//    엔진 로드, 입력/출력 메모리 관리, TensorRT enqueue, 후처리 담당.

// 3. yolo_preprocess.cu
//    yolo_trt_engine.cpp 내부 Preprocess()에서 호출됨.
//    YOLO 입력용으로 BGR -> RGB, HWC -> CHW, float [0,1], letterbox padding
//    수행.

// 4. yolo_trt_engine.cpp
//    TensorRT 추론 후 output [x1,y1,x2,y2,conf,class]를 원본 이미지 좌표 bbox로
//    복원.

// 5-a. line_point_extractor.cpp
//      YOLO detection 중 line class만 골라 여러 bbox 중심점 `(u,v)` 추출.

// 5-b. object_target_extractor.cpp
//      YOLO detection 중 ball/goal/backboard/hurdle class를 단일 target으로
//      추출. 현재는 object 접근 controller용 skeleton이며
//      bbox/center/confidence/size를 유지함.

// 6. coordinate_rectifier.cpp
//    IMU roll/pitch를 이용해서 line 중심점들과 object 중심점만 보정.
//    object bbox width/height/area는 접근/정지 판단용으로 원본 값을 유지.

// 7. feature_extractor.cpp
//    보정된 line 중심점들만 8D feature로 변환.

// 8. line_perception_node.cpp
//    현재는 line feature 기반 rule 속도 명령을 `/g1_vision/cmd_vel`로 publish.
//    이후 mode manager가 line/object 중 하나의 명령만 선택하도록 확장 예정.

// 터미널1: ros2 run my_cv val_image_publisher
// 터미널2: ros2 run vision line_perception_node

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/parameter_map.hpp>
#include <rclcpp/rclcpp.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <sys/utsname.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vision/coordinate_rectifier.hpp"
#include "vision/feature_extractor.hpp"
#include "vision/line_point_extractor.hpp"
#include "vision/object_target_extractor.hpp"
#include "vision/yolo_trt_engine.hpp"

namespace vision {

namespace {

bool IsJetsonTarget() {
  struct utsname info {};
  if (uname(&info) != 0) {
    return false;
  }
  const std::string machine(info.machine);
  return machine == "aarch64" || machine == "arm64";
}

std::string RuntimeConfigPath() {
  if (const char *env_path = std::getenv("YOLO26_RUNTIME_CONFIG"); env_path != nullptr && env_path[0] != '\0') {
    return std::string(env_path);
  }
  return ament_index_cpp::get_package_share_directory("vision") + "/config/yolo26_runtime.yaml";
}

std::string ReadRuntimeString(const std::string &param_name) {
  try {
    const auto map = rclcpp::parameter_map_from_yaml_file(RuntimeConfigPath());
    for (const auto &node_name : {"/yolo26_runtime", "yolo26_runtime"}) {
      const auto it = map.find(node_name);
      if (it == map.end()) {
        continue;
      }
      for (const auto &param : it->second) {
        if (param.get_name() == param_name) {
          return param.as_string();
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[WARN] Failed to load YOLO26 runtime config: " << e.what() << std::endl;
  }
  return {};
}

std::string DefaultYolo26EnginePath() {
  if (const char *env_path = std::getenv("YOLO26_ENGINE_PATH"); env_path != nullptr && env_path[0] != '\0') {
    return std::string(env_path);
  }

  return ReadRuntimeString(IsJetsonTarget() ? "jetson_engine_path" : "pc_engine_path");
}

} // namespace

class LinePerceptionNode final : public rclcpp::Node {
public:
  LinePerceptionNode() : rclcpp::Node("line_perception_node") {
    declare_parameter<std::string>("image_topic", "/camera/color/image_raw"); // 입력 이미지 토픽
    declare_parameter<std::string>("imu_topic", "/camera/imu_tilt");          // IMU roll/pitch 토픽
    declare_parameter<std::string>("prev_cmd_topic", "/g1_vision/cmd_vel");   // 이전 속도 참조 토픽
    declare_parameter<std::string>("cmd_topic", "/g1_vision/cmd_vel");        // 최종 속도 publish 토픽
    declare_parameter<std::string>("engine_path", DefaultYolo26EnginePath()); // TensorRT engine 경로
    declare_parameter<int>("line_class_id", 0);                               // line YOLO class id
    declare_parameter<int>("ball_class_id", 1);                               // ball YOLO class id
    declare_parameter<int>("goal_class_id", 2);                               // goal YOLO class id
    declare_parameter<int>("backboard_class_id", 3);                          // backboard YOLO class id
    declare_parameter<int>("hurdle_class_id", 4);                             // hurdle YOLO class id
    declare_parameter<double>("conf_thres", 0.60);                            // YOLO 1차 confidence threshold
    declare_parameter<double>("ball_conf_thres", 0.60);                       // ball target 2차 confidence threshold
    declare_parameter<double>("goal_conf_thres", 0.60);                       // goal target 2차 confidence threshold
    declare_parameter<double>("backboard_conf_thres", 0.60);      // backboard target 2차 confidence threshold
    declare_parameter<double>("hurdle_conf_thres", 0.60);         // hurdle target 2차 confidence threshold
    declare_parameter<int>("object_min_box_width", 2);            // object 최소 bbox width
    declare_parameter<int>("object_min_box_height", 2);           // object 최소 bbox height
    declare_parameter<double>("fx", 600.0);                       // camera intrinsic fx
    declare_parameter<double>("fy", 600.0);                       // camera intrinsic fy
    declare_parameter<double>("cx", 320.0);                       // camera intrinsic cx
    declare_parameter<double>("cy", 240.0);                       // camera intrinsic cy
    declare_parameter<bool>("use_imu_rectification", true);       // IMU 기반 픽셀 보정 사용 여부
    declare_parameter<bool>("assume_zero_imu", false);            // roll/pitch를 0으로 가정
    declare_parameter<double>("imu_abs_limit_deg", 45.0);         // IMU roll/pitch clamp 각도
    declare_parameter<bool>("show_debug_view", true);             // OpenCV debug window 표시
    declare_parameter<double>("debug_view_hz", 15.0);             // OpenCV debug window 최대 갱신 주기
    declare_parameter<double>("inference_hz", 15.0);              // YOLO 추론 주기
    declare_parameter<int>("max_centers", 8);                     // line feature 최대 점 개수
    declare_parameter<double>("lookahead_delta_v_px", 190.0);     // near point 위쪽 lookahead 거리
    declare_parameter<double>("lookahead_alpha_normal", 0.70);    // normal 상태 lookahead 반영 비율
    declare_parameter<double>("lookahead_alpha_recovery", 0.20);  // recovery 상태 lookahead 반영 비율
    declare_parameter<double>("recover_enter_nvis", 2.0);         // recovery 진입 visible 점 개수
    declare_parameter<double>("recover_exit_nvis", 3.0);          // recovery 탈출 visible 점 개수
    declare_parameter<double>("recover_enter_u", 0.70);           // recovery 진입 lateral error
    declare_parameter<double>("recover_exit_u", 0.35);            // recovery 탈출 lateral error
    declare_parameter<double>("vx_prev_min", 0.0);                // 이전 vx clamp 최소값
    declare_parameter<double>("vx_prev_max", 1.2);                // 이전 vx clamp 최대값
    declare_parameter<double>("wz_prev_min", -1.9);               // 이전 wz clamp 최소값
    declare_parameter<double>("wz_prev_max", 1.9);                // 이전 wz clamp 최대값
    declare_parameter<bool>("enable_rule_controller", true);      // line rule controller publish 여부
    declare_parameter<double>("cmd_vx_min", 0.10);                // 최종 vx 최소값
    declare_parameter<double>("cmd_vx_max", 1.20);                // 최종 vx 최대값
    declare_parameter<double>("cmd_wz_min", -1.90);               // 최종 wz 최소값
    declare_parameter<double>("cmd_wz_max", 1.90);                // 최종 wz 최대값
    declare_parameter<double>("rule_v_base", 0.85);               // line 추종 기본 전진 속도
    declare_parameter<double>("rule_k_u", 3.00);                  // u_err_ctrl 회전 gain
    declare_parameter<double>("rule_k_slope", 3.00);              // slope 회전 gain
    declare_parameter<double>("rule_k_v_u", 0.35);                // lateral error 감속 gain
    declare_parameter<double>("rule_k_v_slope", 0.25);            // slope 감속 gain
    declare_parameter<double>("rule_dv_max", 0.12);               // 프레임당 vx 변화 제한
    declare_parameter<double>("rule_dw_max", 0.40);               // 프레임당 wz 변화 제한
    declare_parameter<double>("rule_recover_vx", 0.12);           // recovery 전진 속도
    declare_parameter<double>("rule_recover_wz", 0.75);           // recovery 회전 속도
    declare_parameter<double>("rule_low_visible_n", 2.0);         // low visibility visible 점 기준
    declare_parameter<double>("rule_no_visible_n", 0.5);          // no visibility visible 점 기준
    declare_parameter<double>("rule_low_visible_vx", 0.18);       // low visibility 전진 속도
    declare_parameter<double>("rule_no_visible_vx", 0.10);        // no visibility 전진 속도
    declare_parameter<double>("rule_low_visible_wz_decay", 0.90); // low visibility 이전 wz 유지 비율
    declare_parameter<double>("rule_no_visible_wz_decay", 0.95);  // no visibility 이전 wz 유지 비율

    image_topic_ = get_parameter("image_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    prev_cmd_topic_ = get_parameter("prev_cmd_topic").as_string();
    cmd_topic_ = get_parameter("cmd_topic").as_string();
    engine_path_ = get_parameter("engine_path").as_string();
    line_class_id_ = static_cast<int>(get_parameter("line_class_id").as_int());
    ball_class_id_ = static_cast<int>(get_parameter("ball_class_id").as_int());
    goal_class_id_ = static_cast<int>(get_parameter("goal_class_id").as_int());
    backboard_class_id_ = static_cast<int>(get_parameter("backboard_class_id").as_int());
    hurdle_class_id_ = static_cast<int>(get_parameter("hurdle_class_id").as_int());
    conf_thres_ = static_cast<float>(get_parameter("conf_thres").as_double());
    ball_conf_thres_ = static_cast<float>(get_parameter("ball_conf_thres").as_double());
    goal_conf_thres_ = static_cast<float>(get_parameter("goal_conf_thres").as_double());
    backboard_conf_thres_ = static_cast<float>(get_parameter("backboard_conf_thres").as_double());
    hurdle_conf_thres_ = static_cast<float>(get_parameter("hurdle_conf_thres").as_double());
    use_imu_rectification_ = get_parameter("use_imu_rectification").as_bool();
    assume_zero_imu_ = get_parameter("assume_zero_imu").as_bool();
    imu_abs_limit_rad_ = Deg2Rad(get_parameter("imu_abs_limit_deg").as_double());
    show_debug_view_ = get_parameter("show_debug_view").as_bool();
    debug_view_hz_ = get_parameter("debug_view_hz").as_double();
    inference_hz_ = get_parameter("inference_hz").as_double();
    vx_prev_min_ = get_parameter("vx_prev_min").as_double();
    vx_prev_max_ = get_parameter("vx_prev_max").as_double();
    wz_prev_min_ = get_parameter("wz_prev_min").as_double();
    wz_prev_max_ = get_parameter("wz_prev_max").as_double();
    enable_rule_controller_ = get_parameter("enable_rule_controller").as_bool();
    cmd_vx_min_ = get_parameter("cmd_vx_min").as_double();
    cmd_vx_max_ = get_parameter("cmd_vx_max").as_double();
    cmd_wz_min_ = get_parameter("cmd_wz_min").as_double();
    cmd_wz_max_ = get_parameter("cmd_wz_max").as_double();
    rule_v_base_ = get_parameter("rule_v_base").as_double();
    rule_k_u_ = get_parameter("rule_k_u").as_double();
    rule_k_slope_ = get_parameter("rule_k_slope").as_double();
    rule_k_v_u_ = get_parameter("rule_k_v_u").as_double();
    rule_k_v_slope_ = get_parameter("rule_k_v_slope").as_double();
    rule_dv_max_ = get_parameter("rule_dv_max").as_double();
    rule_dw_max_ = get_parameter("rule_dw_max").as_double();
    rule_recover_vx_ = get_parameter("rule_recover_vx").as_double();
    rule_recover_wz_ = get_parameter("rule_recover_wz").as_double();
    rule_low_visible_n_ = get_parameter("rule_low_visible_n").as_double();
    rule_no_visible_n_ = get_parameter("rule_no_visible_n").as_double();
    rule_low_visible_vx_ = get_parameter("rule_low_visible_vx").as_double();
    rule_no_visible_vx_ = get_parameter("rule_no_visible_vx").as_double();
    rule_low_visible_wz_decay_ = get_parameter("rule_low_visible_wz_decay").as_double();
    rule_no_visible_wz_decay_ = get_parameter("rule_no_visible_wz_decay").as_double();

    const double fx = get_parameter("fx").as_double();
    const double fy = get_parameter("fy").as_double();
    const double cx = get_parameter("cx").as_double();
    const double cy = get_parameter("cy").as_double();

    feature_cfg_.max_centers = static_cast<int>(get_parameter("max_centers").as_int());
    feature_cfg_.image_center_u = cx;
    feature_cfg_.lookahead_delta_v_px = get_parameter("lookahead_delta_v_px").as_double();
    feature_cfg_.lookahead_alpha_normal = get_parameter("lookahead_alpha_normal").as_double();
    feature_cfg_.lookahead_alpha_recovery = get_parameter("lookahead_alpha_recovery").as_double();
    feature_cfg_.recover_enter_nvis = get_parameter("recover_enter_nvis").as_double();
    feature_cfg_.recover_exit_nvis = get_parameter("recover_exit_nvis").as_double();
    feature_cfg_.recover_enter_u = get_parameter("recover_enter_u").as_double();
    feature_cfg_.recover_exit_u = get_parameter("recover_exit_u").as_double();

    if (engine_path_.empty()) {
      throw std::runtime_error("YOLO TensorRT engine 경로(engine_path)가 비어 있습니다.");
    }

    yolo_ = std::make_unique<YoloTrtEngine>(engine_path_, 640, 640, conf_thres_);
    line_point_extractor_ = std::make_unique<LinePointExtractor>(line_class_id_, conf_thres_);

    ObjectTargetExtractor::Config object_target_cfg;
    object_target_cfg.ball_class_id = ball_class_id_;
    object_target_cfg.goal_class_id = goal_class_id_;
    object_target_cfg.backboard_class_id = backboard_class_id_;
    object_target_cfg.hurdle_class_id = hurdle_class_id_;
    object_target_cfg.ball_conf_thres = ball_conf_thres_;
    object_target_cfg.goal_conf_thres = goal_conf_thres_;
    object_target_cfg.backboard_conf_thres = backboard_conf_thres_;
    object_target_cfg.hurdle_conf_thres = hurdle_conf_thres_;
    object_target_cfg.min_box_width = static_cast<int>(get_parameter("object_min_box_width").as_int());
    object_target_cfg.min_box_height = static_cast<int>(get_parameter("object_min_box_height").as_int());
    object_target_extractor_ = std::make_unique<ObjectTargetExtractor>(object_target_cfg);

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
        image_topic_, rclcpp::SensorDataQoS(), std::bind(&LinePerceptionNode::OnImage, this, std::placeholders::_1));

    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    imu_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    imu_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    imu_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        imu_topic_, imu_qos, std::bind(&LinePerceptionNode::OnImu, this, std::placeholders::_1));

    prev_cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        prev_cmd_topic_, rclcpp::QoS(5), std::bind(&LinePerceptionNode::OnPrevCmd, this, std::placeholders::_1));

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic_, rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "vision node started.");
    RCLCPP_INFO(get_logger(), "  image_topic    : %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  imu_topic      : %s", imu_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  prev_cmd_topic : %s", prev_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  cmd_topic      : %s", cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  engine_path    : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  line_class_id  : %d", line_class_id_);
    RCLCPP_INFO(get_logger(), "  ball_class_id  : %d", ball_class_id_);
    RCLCPP_INFO(get_logger(), "  goal_class_id  : %d", goal_class_id_);
    RCLCPP_INFO(get_logger(), "  backboard_cls  : %d", backboard_class_id_);
    RCLCPP_INFO(get_logger(), "  hurdle_class_id: %d", hurdle_class_id_);
    RCLCPP_INFO(get_logger(), "  ball_conf_thres: %.2f", ball_conf_thres_);
    RCLCPP_INFO(get_logger(), "  goal_conf_thres: %.2f", goal_conf_thres_);
    RCLCPP_INFO(get_logger(), "  backboard_conf : %.2f", backboard_conf_thres_);
    RCLCPP_INFO(get_logger(), "  hurdle_conf_thr: %.2f", hurdle_conf_thres_);
    RCLCPP_INFO(get_logger(), "  inference_hz   : %.2f", inference_hz_);
    RCLCPP_INFO(get_logger(), "  debug_view_hz  : %.2f", debug_view_hz_);
    RCLCPP_INFO(get_logger(), "  imu_rectify    : %s", use_imu_rectification_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  zero_imu       : %s", assume_zero_imu_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  rule_ctrl      : %s", enable_rule_controller_ ? "true" : "false");
    if (ball_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "ball_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  ball_conf_thres_, conf_thres_);
    }
    if (goal_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "goal_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  goal_conf_thres_, conf_thres_);
    }
    if (backboard_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "backboard_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  backboard_conf_thres_, conf_thres_);
    }
    if (hurdle_conf_thres_ < conf_thres_) {
      RCLCPP_WARN(get_logger(),
                  "hurdle_conf_thres(%.2f) < conf_thres(%.2f), but YOLO "
                  "engine output is already filtered by conf_thres.",
                  hurdle_conf_thres_, conf_thres_);
    }

    RCLCPP_INFO(get_logger(), "[SETUP] image subscription created (SensorDataQoS)");
    RCLCPP_INFO(get_logger(), "[SETUP] imu subscription created");
    RCLCPP_INFO(get_logger(), "[SETUP] cmd publisher created");

    if (show_debug_view_) {
      try {
        cv::namedWindow(kDebugWindowName, cv::WINDOW_NORMAL);
        RCLCPP_INFO(get_logger(), "[SETUP] debug window created: %s", kDebugWindowName);
      } catch (const cv::Exception &e) {
        show_debug_view_ = false;
        RCLCPP_WARN(get_logger(), "[SETUP] debug window disabled: %s", e.what());
      }
    }

    // ---- heartbeat(1Hz) ----
    hb_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {
      RCLCPP_INFO(get_logger(), "[HB] frames=%zu imu=%zu pub=%zu (last_img_stamp=%.9f)", frames_count_, imu_count_,
                  pub_count_, last_img_stamp_sec_);
    });

    if (inference_hz_ > 0.0) {
      const auto period =
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / inference_hz_));
      inference_timer_ = create_wall_timer(period, [this]() { ProcessLatestImage(); });
    }

    // ---- PERF(1초 갱신) 초기화: "오버레이만" ----
    last_report_time_ = this->get_clock()->now();
    perf_frame_count_ = 0;
    perf_infer_time_sec_ = 0.0;
    perf_loop_time_sec_ = 0.0;
    perf_text_ = "INFER: -- ms | LOOP: -- ms | FPS: --";
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
  static double Deg2Rad(const double deg) {
    return deg * M_PI / 180.0;
  }

  const char *ClassLabel(int class_id) const {
    if (class_id == line_class_id_) {
      return "line";
    }
    if (class_id == ball_class_id_) {
      return "ball";
    }
    if (class_id == goal_class_id_) {
      return "goal";
    }
    if (class_id == backboard_class_id_) {
      return "backboard";
    }
    if (class_id == hurdle_class_id_) {
      return "hurdle";
    }
    return "unknown";
  }

  bool ShouldDrawDebugView(const std::chrono::steady_clock::time_point &now) {
    if (!show_debug_view_) {
      return false;
    }
    if (debug_view_hz_ <= 0.0 || last_debug_view_time_ == std::chrono::steady_clock::time_point{}) {
      last_debug_view_time_ = now;
      return true;
    }

    const auto min_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / debug_view_hz_));
    if (now - last_debug_view_time_ < min_interval) {
      return false;
    }

    last_debug_view_time_ = now;
    return true;
  }
  static double Clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
  }
  static double Sign(double v) {
    if (v > 0.0)
      return 1.0;
    if (v < 0.0)
      return -1.0;
    return 0.0;
  }

  void OnPrevCmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
    ++prev_cmd_count_;
    vx_prev_ = Clamp(static_cast<double>(msg->linear.x), vx_prev_min_, vx_prev_max_);
    wz_prev_ = Clamp(static_cast<double>(msg->angular.z), wz_prev_min_, wz_prev_max_);
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

    RCLCPP_DEBUG(get_logger(), "[IMU] recv roll=%.5f pitch=%.5f (clamped) | imu_ready=%d", last_roll_rad_,
                 last_pitch_rad_, imu_ready_ ? 1 : 0);
  }

  void DrawDetections(cv::Mat &vis, const std::vector<Detection> &dets,
                      const std::optional<ObjectTargetExtractor::Target> &ball_target,
                      const std::optional<ObjectTargetExtractor::Target> &goal_target,
                      const std::optional<ObjectTargetExtractor::Target> &backboard_target,
                      const std::optional<ObjectTargetExtractor::Target> &hurdle_target) {
    for (const auto &det : dets) {
      if (det.confidence < conf_thres_) {
        continue;
      }

      const cv::Rect box = det.box;
      cv::Rect clipped = box & cv::Rect(0, 0, vis.cols, vis.rows);
      if (clipped.width <= 0 || clipped.height <= 0) {
        continue;
      }

      const int cx = clipped.x + clipped.width / 2;
      const int cy = clipped.y + clipped.height / 2;

      cv::Scalar box_color(160, 160, 160);
      cv::Scalar center_color(80, 80, 80);
      if (det.class_id == line_class_id_) {
        box_color = cv::Scalar(0, 255, 0);
        center_color = cv::Scalar(0, 0, 255);
      } else if (det.class_id == ball_class_id_) {
        box_color = cv::Scalar(0, 165, 255);
        center_color = cv::Scalar(255, 0, 0);
      } else if (det.class_id == goal_class_id_) {
        box_color = cv::Scalar(255, 255, 0);
        center_color = cv::Scalar(0, 255, 255);
      } else if (det.class_id == backboard_class_id_) {
        box_color = cv::Scalar(255, 128, 0);
        center_color = cv::Scalar(0, 128, 255);
      } else if (det.class_id == hurdle_class_id_) {
        box_color = cv::Scalar(255, 0, 255);
        center_color = cv::Scalar(255, 255, 255);
      }

      cv::rectangle(vis, clipped, box_color, 2);
      cv::circle(vis, cv::Point(cx, cy), 3, center_color, -1);

      char buf[40];
      std::snprintf(buf, sizeof(buf), "%s %.2f", ClassLabel(det.class_id), det.confidence);

      int tx = clipped.x;
      int ty = clipped.y - 6;
      if (ty < 12)
        ty = clipped.y + 16;

      cv::putText(vis, buf, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2);
    }

    const auto draw_rectified_center = [&vis](const std::optional<ObjectTargetExtractor::Target> &target,
                                              const cv::Scalar &color) {
      if (!target || !target->center_rectified) {
        return;
      }
      const cv::Point2f p = target->rectified_center_px;
      if (std::isfinite(p.x) && std::isfinite(p.y)) {
        cv::circle(vis, cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))), 5, color, 2);
      }
    };

    draw_rectified_center(ball_target, cv::Scalar(255, 255, 0));
    draw_rectified_center(goal_target, cv::Scalar(0, 255, 255));
    draw_rectified_center(backboard_target, cv::Scalar(0, 128, 255));
    draw_rectified_center(hurdle_target, cv::Scalar(255, 255, 255));
  }

  void DrawTargetSummary(cv::Mat &vis, const char *name, const std::optional<ObjectTargetExtractor::Target> &target,
                         int y, const cv::Scalar &color) {
    if (!target) {
      return;
    }

    const cv::Point2f p = target->center_rectified ? target->rectified_center_px : target->center_px;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: conf=%.2f u=%.1f v=%.1f h=%.0f area=%.0f", name, target->confidence, p.x, p.y,
                  target->height_px, target->area_px);
    cv::putText(vis, buf, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
  }

  // PERF: 1초에 1번만 문자열 갱신 (로그로는 절대 안 찍음)
  void UpdatePerfOverlayOncePerSecond(double infer_time_sec, double loop_time_sec) {
    const rclcpp::Time now = this->get_clock()->now();
    const rclcpp::Duration elapsed = now - last_report_time_;

    perf_frame_count_ += 1;
    perf_infer_time_sec_ += infer_time_sec;
    perf_loop_time_sec_ += loop_time_sec;

    if (elapsed.seconds() >= 1.0) {
      const int denom = std::max(1, perf_frame_count_);
      const double avg_infer_ms = (perf_infer_time_sec_ / static_cast<double>(denom)) * 1000.0;
      const double avg_loop_ms = (perf_loop_time_sec_ / static_cast<double>(denom)) * 1000.0;
      const double fps = static_cast<double>(perf_frame_count_) / elapsed.seconds();

      last_report_time_ = now;
      perf_frame_count_ = 0;
      perf_infer_time_sec_ = 0.0;
      perf_loop_time_sec_ = 0.0;

      char buf[128];
      std::snprintf(buf, sizeof(buf), "INFER: %.2fms | LOOP: %.2fms | FPS: %.2f", avg_infer_ms, avg_loop_ms, fps);
      perf_text_ = buf;
    }
  }

  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    ++frames_count_;
    last_img_stamp_sec_ =
        static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

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
    RCLCPP_DEBUG(get_logger(), "[IMU] snapshot roll=%.5f pitch=%.5f | imu_ready=%d", roll, pitch,
                 imu_ready_snapshot ? 1 : 0);

    // 3) YOLO 추론
    std::vector<Detection> dets;

    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = yolo_->Infer(bgr, dets);
    const auto t2 = std::chrono::steady_clock::now();

    const auto dt_yolo_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (!ok) {
      RCLCPP_ERROR(get_logger(), "[YOLO] Infer failed -> skip remaining stages");
      return;
    }

    // 4) Detection -> line centers + object targets
    const auto t3 = std::chrono::steady_clock::now();
    std::vector<cv::Point2f> centers_px = line_point_extractor_->ExtractCenters(dets);
    std::optional<ObjectTargetExtractor::Target> ball_target = object_target_extractor_->ExtractBall(dets);
    std::optional<ObjectTargetExtractor::Target> goal_target = object_target_extractor_->ExtractGoal(dets);
    std::optional<ObjectTargetExtractor::Target> backboard_target = object_target_extractor_->ExtractBackboard(dets);
    std::optional<ObjectTargetExtractor::Target> hurdle_target = object_target_extractor_->ExtractHurdle(dets);
    const auto t4 = std::chrono::steady_clock::now();
    const auto dt_pts_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // 5) 좌표 보정: line은 전체 중심점, object는 중심점만 보정한다.
    std::vector<cv::Point2f> centers_rect_px = centers_px;
    const auto t5 = std::chrono::steady_clock::now();
    if (assume_zero_imu_) {
      roll = 0.0;
      pitch = 0.0;
      imu_ready_snapshot = true;
    }

    if (use_imu_rectification_ && imu_ready_snapshot) {
      centers_rect_px = rectifier_->RectifyPixelPoints(centers_px, roll, pitch);
      const auto rectify_target_center = [this, roll, pitch](std::optional<ObjectTargetExtractor::Target> &target) {
        if (!target) {
          return;
        }
        const std::vector<cv::Point2f> rectified_center =
            rectifier_->RectifyPixelPoints({target->center_px}, roll, pitch);
        if (!rectified_center.empty()) {
          target->rectified_center_px = rectified_center.front();
          target->center_rectified = true;
        }
      };

      rectify_target_center(ball_target);
      rectify_target_center(goal_target);
      rectify_target_center(backboard_target);
      rectify_target_center(hurdle_target);
    }
    const auto t6 = std::chrono::steady_clock::now();
    const auto dt_rec_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    // 6) line feature 계산 후 rule 기반 속도 명령 publish.
    //    ball target은 다음 mode/controller 단계에서 사용할 skeleton 상태로만
    //    유지.
    const auto t7 = std::chrono::steady_clock::now();
    const auto feats =
        feature_extractor_->Compute(centers_rect_px, bgr.size(), in_recovery_, vx_prev_, wz_prev_, feature_cfg_);
    in_recovery_ = feats.in_recovery > 0.5;
    const auto t8 = std::chrono::steady_clock::now();
    const auto dt_feat_us = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();

    if (enable_rule_controller_) {
      PublishRuleCommand(feats);
    }

    // ---- 시각화 ----
    if (ShouldDrawDebugView(std::chrono::steady_clock::now())) {
      cv::Mat vis = bgr.clone();
      DrawDetections(vis, dets, ball_target, goal_target, backboard_target, hurdle_target);

      // object target 요약(매 프레임)
      DrawTargetSummary(vis, "BALL", ball_target, 150, cv::Scalar(0, 165, 255));
      DrawTargetSummary(vis, "GOAL", goal_target, 180, cv::Scalar(255, 255, 0));
      DrawTargetSummary(vis, "BACKBOARD", backboard_target, 210, cv::Scalar(255, 128, 0));
      DrawTargetSummary(vis, "HURDLE", hurdle_target, 240, cv::Scalar(255, 0, 255));

      // perf_text_ (1초에 1번만 갱신되는 문자열)
      cv::putText(vis, perf_text_, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

      // ===== IMU overlay =====
      {
        char buf[128];

        if (imu_ready_snapshot) {
          std::snprintf(buf, sizeof(buf),
                        "IMU: TRUE | roll=%.3f rad (%.1f deg) | pitch=%.3f rad "
                        "(%.1f deg)",
                        roll, roll * 180.0 / M_PI, pitch, pitch * 180.0 / M_PI);
        } else {
          std::snprintf(buf, sizeof(buf), "IMU: FALSE | roll=NONE | pitch=NONE");
        }

        cv::putText(vis, buf, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
      }

      // ===== Rule command overlay =====
      {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "CMD: vx=%.3f m/s | wz=%.3f rad/s", vx_prev_, wz_prev_);
        cv::putText(vis, buf, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2);
      }

      try {
        cv::imshow(kDebugWindowName, vis);
        cv::waitKey(1);
      } catch (const cv::Exception &e) {
        show_debug_view_ = false;
        RCLCPP_WARN(get_logger(), "[VIEW] debug window disabled: %s", e.what());
      }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const auto dt_loop_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t0).count();

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "[TIME] yolo=%.2fms pts=%.2fms rect=%.2fms "
                         "feat=%.2fms infer=%.2fms loop=%.2fms n=%.0f rec=%.0f "
                         "ball=%d goal=%d backboard=%d hurdle=%d",
                         static_cast<double>(dt_yolo_us) / 1000.0, static_cast<double>(dt_pts_us) / 1000.0,
                         static_cast<double>(dt_rec_us) / 1000.0, static_cast<double>(dt_feat_us) / 1000.0,
                         static_cast<double>(dt_yolo_us) / 1000.0, static_cast<double>(dt_loop_us) / 1000.0,
                         feats.n_visible, feats.in_recovery, ball_target ? 1 : 0, goal_target ? 1 : 0,
                         backboard_target ? 1 : 0, hurdle_target ? 1 : 0);

    // ===== PERF overlay 업데이트(1초에 1번만 문자열 갱신) =====
    UpdatePerfOverlayOncePerSecond(static_cast<double>(dt_yolo_us) * 1e-6, static_cast<double>(dt_loop_us) * 1e-6);
  }

  void PublishRuleCommand(const FeatureExtractor::Features &feats) {
    const double u_err_ctrl = feats.u_err_ctrl;
    const double slope = feats.slope;
    const double n_visible = feats.n_visible;
    const bool in_recovery = feats.in_recovery > 0.5;

    double w_nom = -rule_k_u_ * u_err_ctrl - rule_k_slope_ * slope;
    double v_nom = rule_v_base_ - rule_k_v_u_ * std::abs(u_err_ctrl) - rule_k_v_slope_ * std::abs(slope);

    const bool no_vis = n_visible < rule_no_visible_n_;
    const bool low_vis = n_visible < rule_low_visible_n_;
    if (low_vis) {
      w_nom = rule_low_visible_wz_decay_ * rule_last_wz_;
      v_nom = rule_low_visible_vx_;
    }

    double turn_sign = Sign(w_nom);
    if (turn_sign == 0.0) {
      turn_sign = Sign(rule_last_wz_);
    }
    if (turn_sign == 0.0) {
      turn_sign = 1.0;
    }

    double v_raw = in_recovery ? rule_recover_vx_ : v_nom;
    double w_raw = in_recovery ? rule_recover_wz_ * turn_sign : w_nom;

    if (no_vis) {
      v_raw = rule_no_visible_vx_;
      w_raw = rule_no_visible_wz_decay_ * rule_last_wz_;
    }

    double vx = Clamp(v_raw, cmd_vx_min_, cmd_vx_max_);
    double wz = Clamp(w_raw, cmd_wz_min_, cmd_wz_max_);

    vx = Clamp(vx, rule_last_vx_ - rule_dv_max_, rule_last_vx_ + rule_dv_max_);
    wz = Clamp(wz, rule_last_wz_ - rule_dw_max_, rule_last_wz_ + rule_dw_max_);

    rule_last_vx_ = vx;
    rule_last_wz_ = wz;
    vx_prev_ = vx;
    wz_prev_ = wz;

    geometry_msgs::msg::Twist out;
    out.linear.x = vx;
    out.angular.z = wz;
    cmd_pub_->publish(out);
    ++pub_count_;

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "[RULE] %s | vx=%.3f wz=%.3f u_ctrl=%.3f slope=%.3f n=%.0f rec=%d", cmd_topic_.c_str(), vx, wz,
                         u_err_ctrl, slope, n_visible, in_recovery ? 1 : 0);
  }

private:
  // ROS Sub/Pub
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr prev_cmd_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr hb_timer_;
  rclcpp::TimerBase::SharedPtr inference_timer_;

  // 모듈
  std::unique_ptr<YoloTrtEngine> yolo_;
  std::unique_ptr<LinePointExtractor> line_point_extractor_;
  std::unique_ptr<ObjectTargetExtractor> object_target_extractor_;
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

  bool enable_rule_controller_{true};
  double cmd_vx_min_{0.10};
  double cmd_vx_max_{1.20};
  double cmd_wz_min_{-1.90};
  double cmd_wz_max_{1.90};
  double rule_v_base_{0.85};
  double rule_k_u_{3.00};
  double rule_k_slope_{3.00};
  double rule_k_v_u_{0.35};
  double rule_k_v_slope_{0.25};
  double rule_dv_max_{0.12};
  double rule_dw_max_{0.40};
  double rule_recover_vx_{0.12};
  double rule_recover_wz_{0.75};
  double rule_low_visible_n_{2.0};
  double rule_no_visible_n_{0.5};
  double rule_low_visible_vx_{0.18};
  double rule_no_visible_vx_{0.10};
  double rule_low_visible_wz_decay_{0.90};
  double rule_no_visible_wz_decay_{0.95};
  double rule_last_vx_{0.0};
  double rule_last_wz_{0.0};

  // 디버그 카운터
  size_t frames_count_{0};
  size_t imu_count_{0};
  size_t pub_count_{0};
  size_t prev_cmd_count_{0};
  double last_img_stamp_sec_{0.0};

  // ---- PERF 오버레이(1초 갱신) ----
  rclcpp::Time last_report_time_{0, 0, RCL_ROS_TIME};
  int perf_frame_count_{0};
  double perf_infer_time_sec_{0.0};
  double perf_loop_time_sec_{0.0};
  std::string perf_text_;

  // ---- 시각화/필터 설정(멤버) ----
  int line_class_id_{0};
  int ball_class_id_{1};
  int goal_class_id_{2};
  int backboard_class_id_{3};
  int hurdle_class_id_{4};
  float conf_thres_{0.60f};
  float ball_conf_thres_{0.60f};
  float goal_conf_thres_{0.60f};
  float backboard_conf_thres_{0.60f};
  float hurdle_conf_thres_{0.60f};
  bool show_debug_view_{false};
  double debug_view_hz_{5.0};
  std::chrono::steady_clock::time_point last_debug_view_time_;
  double inference_hz_{5.0};

  std::string image_topic_;
  std::string imu_topic_;
  std::string prev_cmd_topic_;
  std::string cmd_topic_;
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
