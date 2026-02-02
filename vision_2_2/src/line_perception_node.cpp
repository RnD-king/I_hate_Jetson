#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vision/yolo_trt_engine.hpp"
#include "vision/line_point_extractor.hpp"
#include "vision/coordinate_rectifier.hpp"
#include "vision/feature_extractor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vision {

class LinePerceptionNode final : public rclcpp::Node {
public:
  LinePerceptionNode()
  : rclcpp::Node("line_perception_node")
  {
    image_topic_    = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    features_topic_ = declare_parameter<std::string>("features_topic", "/line_features");
    engine_path_    = declare_parameter<std::string>("engine_path",
                      "/home/rnd/rnd/yolo/v8n/best_8n_fp16.engine");

    line_class_id_  = declare_parameter<int>("line_class_id", 0);
    conf_thres_     = declare_parameter<double>("conf_thres", 0.25);

    fx_ = declare_parameter<double>("fx", 600.0);
    fy_ = declare_parameter<double>("fy", 600.0);
    cx_ = declare_parameter<double>("cx", 320.0);
    cy_ = declare_parameter<double>("cy", 240.0);

    yolo_ = std::make_unique<YoloTrtEngine>(engine_path_);

    point_extractor_ = std::make_unique<LinePointExtractor>(
        line_class_id_, static_cast<float>(conf_thres_));

    CoordinateRectifier::Intrinsics K{};
    K.fx = fx_; K.fy = fy_; K.cx = cx_; K.cy = cy_;
    rectifier_ = std::make_unique<CoordinateRectifier>(K);

    feature_extractor_ = std::make_unique<FeatureExtractor>();

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&LinePerceptionNode::OnImage, this, std::placeholders::_1));

    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      features_topic_, rclcpp::QoS(10));

    imu_ready_ = false;

    RCLCPP_INFO(get_logger(), "vision/line_perception_node started.");
    RCLCPP_INFO(get_logger(), "  image_topic   : %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  features_topic: %s", features_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  engine_path   : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  imu_ready_    : false (fixed)");
  }

private:
  void OnImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      // (1) 메시지 메타데이터 로그/검증
      RCLCPP_INFO(get_logger(),
        "Image msg: enc=%s w=%u h=%u step=%u data=%zu",
        msg->encoding.c_str(), msg->width, msg->height, msg->step, msg->data.size());

      if (msg->width == 0 || msg->height == 0 || msg->step == 0) {
        RCLCPP_ERROR(get_logger(), "Invalid image meta (w/h/step is 0).");
        return;
      }
      const size_t min_bytes = static_cast<size_t>(msg->step) * static_cast<size_t>(msg->height);
      if (msg->data.size() < min_bytes) {
        RCLCPP_ERROR(get_logger(),
          "Invalid image buffer: data.size()=%zu < step*height=%zu",
          msg->data.size(), min_bytes);
        return;
      }

      // (2) cv_bridge: “있는 그대로” 받기 (불필요한 내부 변환 제거)
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);

      const cv::Mat& src = cv_ptr->image;
      RCLCPP_INFO(get_logger(),
        "Mat(src): cols=%d rows=%d type=%d step=%zu continuous=%d",
        src.cols, src.rows, src.type(),
        static_cast<size_t>(src.step[0]), src.isContinuous());

      if (src.empty() || src.cols <= 0 || src.rows <= 0) {
        RCLCPP_ERROR(get_logger(), "Invalid src Mat.");
        return;
      }

      // (3) 엔진 입력은 CV_8UC3(BGR)로 통일
      cv::Mat bgr;

      if (msg->encoding == "rgb8") {
        // RGB -> BGR
        cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
      } else if (msg->encoding == "bgr8") {
        bgr = src;
      } else {
        RCLCPP_ERROR(get_logger(), "Unsupported encoding for this node: %s", msg->encoding.c_str());
        return;
      }

      RCLCPP_INFO(get_logger(),
        "Mat(bgr): cols=%d rows=%d type=%d step=%zu continuous=%d",
        bgr.cols, bgr.rows, bgr.type(),
        static_cast<size_t>(bgr.step[0]), bgr.isContinuous());

      if (bgr.type() != CV_8UC3) {
        RCLCPP_ERROR(get_logger(), "bgr Mat type is not CV_8UC3.");
        return;
      }

      // (4) 연속 메모리 보장 (cudaMemcpyAsync 안전)
      cv::Mat bgr_cont = bgr.isContinuous() ? bgr : bgr.clone();
      if (!bgr_cont.isContinuous()) {
        RCLCPP_ERROR(get_logger(), "bgr_cont is not continuous even after clone.");
        return;
      }

      // (5) inference
      std::vector<Detection> dets;
      if (!yolo_->Infer(bgr_cont, dets)) {
        RCLCPP_WARN(get_logger(), "Infer() returned false.");
        return;
      }

      // (6) points
      std::vector<cv::Point2f> centers_px = point_extractor_->ExtractCenters(dets);

      // (7) optional rectification (fixed false now)
      std::vector<cv::Point2f> centers_used = centers_px;
      if (imu_ready_) {
        centers_used = rectifier_->RectifyPixelPoints(centers_px, 0.0, 0.0);
      }

      // (8) features
      FeatureExtractor::Features feats = feature_extractor_->Compute(centers_used, bgr_cont.size());

      // (9) publish
      std_msgs::msg::Float32MultiArray out;
      out.data = {
        static_cast<float>(feats.slope_mean),
        static_cast<float>(feats.slope_var),
        static_cast<float>(feats.x_mean_norm),
        static_cast<float>(feats.x_var_norm),
        static_cast<float>(feats.y_mean_norm),
        static_cast<float>(feats.y_var_norm),
        static_cast<float>(feats.count_norm),
        static_cast<float>(feats.mean_conf)
      };
      features_pub_->publish(out);

    } catch (const cv::Exception& e) {
      RCLCPP_ERROR(get_logger(), "OpenCV exception in OnImage: %s", e.what());
      return;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "std::exception in OnImage: %s", e.what());
      return;
    } catch (...) {
      RCLCPP_ERROR(get_logger(), "Unknown exception in OnImage.");
      return;
    }
  }

private:
  std::string image_topic_;
  std::string features_topic_;
  std::string engine_path_;
  int line_class_id_{0};
  double conf_thres_{0.25};
  double fx_{600.0}, fy_{600.0}, cx_{320.0}, cy_{240.0};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr features_pub_;

  std::unique_ptr<YoloTrtEngine> yolo_;
  std::unique_ptr<LinePointExtractor> point_extractor_;
  std::unique_ptr<CoordinateRectifier> rectifier_;
  std::unique_ptr<FeatureExtractor> feature_extractor_;

  bool imu_ready_{false};
};

}  // namespace vision

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vision::LinePerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
