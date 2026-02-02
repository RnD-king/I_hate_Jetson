#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "vision/types.hpp"  // Detection만 필요

namespace vision
{

class LinePointExtractor
{
public:
  explicit LinePointExtractor(int line_class_id, float conf_thres);

  // YOLO Detection들 중 "점선 클래스"만 골라 bbox 중심점을 뽑아서 반환
  // - 반환 좌표는 "원본 이미지 픽셀 좌표" (YOLO postprocess가 원본으로 복원했다고 가정)
  // - 결과는 y 기준 내림차순 정렬(아래쪽 점부터) + tie-break(x)로 재현성 보장
  std::vector<cv::Point2f> ExtractCenters(const std::vector<Detection>& dets) const;

private:
  int line_class_id_{0};
  float conf_thres_{0.25f};
};

}  // namespace vision
