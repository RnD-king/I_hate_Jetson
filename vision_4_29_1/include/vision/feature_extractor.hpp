#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace vision
{

class FeatureExtractor
{
public:
  struct Config
  {
    int max_centers{8};
    double image_center_u{-1.0};
    double lookahead_delta_v_px{190.0};
    double lookahead_alpha_normal{0.70};
    double lookahead_alpha_recovery{0.20};
    double recover_enter_nvis{2.0};
    double recover_exit_nvis{3.0};
    double recover_enter_u{0.70};
    double recover_exit_u{0.35};
  };

  struct Features
  {
    // 학습된 g1_vision policy 입력과 동일한 8D 순서.
    double u_err_near{0.0};
    double u_err_lookahead{0.0};
    double u_err_ctrl{0.0};
    double slope{0.0};
    double n_visible{0.0};
    double in_recovery{0.0};
    double vx_prev{0.0};
    double wz_prev{0.0};
  };

  FeatureExtractor() = default;

  // 보정된 픽셀 점들을 학습 코드의 8D feature로 변환한다.
  // 입력 점은 y 내림차순(화면 아래쪽 먼저)으로 재정렬하고 max_centers개만 사용한다.
  Features Compute(
    const std::vector<cv::Point2f>& pts_px,
    const cv::Size& image_size,
    bool previous_in_recovery,
    double vx_prev,
    double wz_prev,
    const Config& cfg) const;

private:
  static constexpr double kEps = 1e-9;
};

}  // namespace vision
