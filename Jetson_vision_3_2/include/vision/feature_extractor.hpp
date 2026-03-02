#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace vision
{

class FeatureExtractor
{
public:
  struct Features
  {
    // [0] slope_mean  : (dx/dy) 평균 (중심점들이 만드는 평균 기울기 proxy)
    // [1] slope_var   : (dx/dy) 분산 (일관성/안정성 지표)
    double slope_mean{0.0};
    double slope_var{0.0};

    // [2] x_mean_norm : x 평균을 [-1, +1]로 정규화 (좌/우 치우침)
    // [3] x_var_norm  : x 분산 정규화 (퍼짐 정도)
    double x_mean_norm{0.0};
    double x_var_norm{0.0};

    // [4] y_mean_norm : y 평균을 [-1, +1]로 정규화 (원근 기반 거리감 proxy로 사용 가능)
    // [5] y_var_norm  : y 분산 정규화
    double y_mean_norm{0.0};
    double y_var_norm{0.0};

    // [6] count_norm  : 검출 개수 정규화(0~1) -> 신뢰도 proxy
    double count_norm{0.0};

    // [7] mean_conf   : 평균 confidence (현재 파이프라인 입력에 없어서 0 고정)
    double mean_conf{0.0};
  };

  FeatureExtractor() = default;

  // 보정된 픽셀 점들을 입력으로 받아 저차원 feature로 요약한다.
  // - 규칙 기반 판단(if문 기반 주행판단)을 하지 않고 통계량만 계산한다.
  // - 점 개수가 프레임마다 달라도, 출력 차원은 항상 고정(Features).
  Features Compute(const std::vector<cv::Point2f>& pts_px, const cv::Size& image_size) const;

private:
  // count 정규화에서 사용할 상한(너무 많아도 1로 포화)
  static constexpr int kMaxCountCap = 20;

  // slope 계산에서 dy가 너무 작을 때(수평에 가까운 경우) 불안정하므로 표본 제외
  static constexpr double kMinAbsDyPx = 1.0;

  // dx/dy 폭주 방지용 클램프(부호 유지)
  static constexpr double kSlopeClampAbs = 100.0;

  // 수치 안전용 epsilon
  static constexpr double kEps = 1e-9;
};

}  // namespace vision