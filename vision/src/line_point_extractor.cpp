#include "vision/line_point_extractor.hpp"

#include <algorithm>

namespace vision
{

LinePointExtractor::LinePointExtractor(int line_class_id, float conf_thres)
: line_class_id_(line_class_id), conf_thres_(conf_thres)
{
}

std::vector<cv::Point2f> LinePointExtractor::ExtractCenters(const std::vector<Detection>& dets) const
{
  std::vector<cv::Point2f> centers;
  centers.reserve(dets.size());

  for (const auto& d : dets)
  {
    // 1) 클래스 필터 (점선 class만)
    if (d.class_id != line_class_id_) continue;

    // 2) confidence 필터
    if (d.confidence < conf_thres_) continue;

    // 3) 박스 유효성(너무 작은 박스는 중심점이 사실상 노이즈일 가능성이 큼)
    const auto& b = d.box;
    if (b.width <= 1 || b.height <= 1) continue;

    // 4) bbox 중심점(픽셀 좌표)
    const float cx = static_cast<float>(b.x) + 0.5f * static_cast<float>(b.width);
    const float cy = static_cast<float>(b.y) + 0.5f * static_cast<float>(b.height);

    centers.emplace_back(cx, cy);
  }

  // 5) y 내림차순 정렬(화면 아래쪽이 더 "가까운" 점)
  //    - 동점(y가 같은 경우)에는 x로 tie-break 해서 결과를 완전 결정적으로 만든다.
  std::sort(centers.begin(), centers.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) {
              if (a.y == b.y) return a.x < b.x;
              return a.y > b.y;
            });

  return centers;
}

}  // namespace vision
