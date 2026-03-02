#include "vision/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vision
{

FeatureExtractor::Features FeatureExtractor::Compute(
  const std::vector<cv::Point2f>& pts_px,
  const cv::Size& image_size) const
{
  Features f{};

  const int W = image_size.width;
  const int H = image_size.height;

  if (W <= 1 || H <= 1)
  {
    // 이미지 크기가 비정상인 경우: 의미 있는 정규화가 불가하므로 0 반환
    return f;
  }

  const int N = static_cast<int>(pts_px.size());
  if (N <= 0)
  {
    // 점이 없으면 전부 0
    f.count_norm = 0.0;
    f.mean_conf = 0.0;
    return f;
  }

  // -----------------------------
  // 1) 점들을 y 기준으로 내림차순 정렬(아래쪽 점이 먼저)
  //    - 입력이 이미 정렬돼 있어도, 여기서 다시 정렬하면 모듈 단독으로도 일관성이 생김
  // -----------------------------
  std::vector<cv::Point2f> pts = pts_px;
  std::stable_sort(pts.begin(), pts.end(),
                   [](const cv::Point2f& a, const cv::Point2f& b) {
                     return a.y > b.y;  // y가 큰 게 아래(가까운 쪽)
                   });

  // -----------------------------
  // 2) x/y 평균 및 분산 (정규화 포함)
  // -----------------------------
  double sum_x = 0.0, sum_y = 0.0;
  for (const auto& p : pts)
  {
    sum_x += static_cast<double>(p.x);
    sum_y += static_cast<double>(p.y);
  }
  const double mean_x = sum_x / static_cast<double>(N);
  const double mean_y = sum_y / static_cast<double>(N);

  double var_x = 0.0, var_y = 0.0;
  for (const auto& p : pts)
  {
    const double dx = static_cast<double>(p.x) - mean_x;
    const double dy = static_cast<double>(p.y) - mean_y;
    var_x += dx * dx;
    var_y += dy * dy;
  }
  var_x /= static_cast<double>(N);
  var_y /= static_cast<double>(N);

  // [-1, +1] 정규화: 화면 중심을 0으로 두고, 반폭/반높이로 나눔
  const double half_w = 0.5 * static_cast<double>(W);
  const double half_h = 0.5 * static_cast<double>(H);

  f.x_mean_norm = (mean_x - half_w) / (half_w + kEps);
  f.y_mean_norm = (mean_y - half_h) / (half_h + kEps);

  // 분산 정규화: (반폭^2), (반높이^2)로 나눔
  f.x_var_norm = var_x / ((half_w * half_w) + kEps);
  f.y_var_norm = var_y / ((half_h * half_h) + kEps);

  // -----------------------------
  // 3) slope (dx/dy) 통계
  //    - dy가 0에 가까우면 폭주(수평선) -> 표본 제외 + 클램프
  // -----------------------------
  if (N >= 2)
  {
    std::vector<double> slopes;
    slopes.reserve(static_cast<size_t>(N - 1));

    for (int i = 1; i < N; ++i)
    {
      const double x0 = static_cast<double>(pts[i - 1].x);
      const double y0 = static_cast<double>(pts[i - 1].y);
      const double x1 = static_cast<double>(pts[i].x);
      const double y1 = static_cast<double>(pts[i].y);

      const double dy = (y1 - y0);
      const double dx = (x1 - x0);

      // dy가 너무 작으면 기울기 정의가 불안정 -> 표본으로 쓰지 않음
      if (std::abs(dy) < kMinAbsDyPx)
        continue;

      // dx/dy
      double s = dx / dy;

      // 폭주 방지: 부호 유지 + [-100, 100] 클램프
      if (s >  kSlopeClampAbs) s =  kSlopeClampAbs;
      if (s < -kSlopeClampAbs) s = -kSlopeClampAbs;

      slopes.push_back(s);
    }

    if (!slopes.empty())
    {
      const double s_sum = std::accumulate(slopes.begin(), slopes.end(), 0.0);
      const double s_mean = s_sum / static_cast<double>(slopes.size());

      double s_var = 0.0;
      for (double s : slopes)
      {
        const double ds = s - s_mean;
        s_var += ds * ds;
      }
      s_var /= static_cast<double>(slopes.size());

      f.slope_mean = s_mean;
      f.slope_var = s_var;
    }
    else
    {
      // slope 표본이 없으면 0으로 둠
      f.slope_mean = 0.0;
      f.slope_var = 0.0;
    }
  }

  // -----------------------------
  // 4) count 정규화 (0~1 포화)
  // -----------------------------
  const int capped = std::min(N, kMaxCountCap);
  f.count_norm = static_cast<double>(capped) / static_cast<double>(kMaxCountCap);

  // -----------------------------
  // 5) mean_conf
  //    - 현재 입력에 confidence가 없으므로 0 고정
  // -----------------------------
  // f.mean_conf = 0.0;

  return f;
}

}  // namespace vision