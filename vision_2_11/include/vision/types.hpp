#pragma once

#include <opencv2/core.hpp>

namespace vision
{

struct Detection
{
  cv::Rect box;
  float confidence{0.0f};
  int class_id{0};
};

}  // namespace vision