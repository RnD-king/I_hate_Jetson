// 특징 벡터 계산 8D.

#include "vision/feature_extractor.hpp"

#include <algorithm>
#include <cmath>

namespace vision {

namespace {

double Clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

} // namespace

FeatureExtractor::Features
FeatureExtractor::Compute(const std::vector<cv::Point2f> &pts_px,
                          const cv::Size &image_size, bool previous_in_recovery,
                          double vx_prev, double wz_prev,
                          const Config &cfg) const {
  Features f{};
  f.vx_prev = vx_prev;
  f.wz_prev = wz_prev;

  const int W = image_size.width;
  const int H = image_size.height;
  if (W <= 1 || H <= 1) {
    f.in_recovery = previous_in_recovery ? 1.0 : 0.0;
    return f;
  }

  std::vector<cv::Point2f> pts;
  pts.reserve(pts_px.size());
  for (const auto &p : pts_px) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      continue;
    const float u = static_cast<float>(
        Clamp(static_cast<double>(p.x), 0.0, static_cast<double>(W - 1)));
    const float v = static_cast<float>(
        Clamp(static_cast<double>(p.y), 0.0, static_cast<double>(H - 1)));
    pts.emplace_back(u, v);
  }

  std::sort(pts.begin(), pts.end(),
            [](const cv::Point2f &a, const cv::Point2f &b) {
              if (a.y == b.y)
                return a.x < b.x;
              return a.y > b.y;
            });

  const int keep_n =
      std::min(std::max(1, cfg.max_centers), static_cast<int>(pts.size()));
  if (static_cast<int>(pts.size()) > keep_n) {
    pts.resize(static_cast<size_t>(keep_n));
  }

  f.n_visible = static_cast<double>(pts.size());
  if (pts.empty()) {
    f.in_recovery = 1.0;
    return f;
  }

  const double cx = (cfg.image_center_u >= 0.0) ? cfg.image_center_u
                                                : 0.5 * static_cast<double>(W);
  const double denom_u = std::max(cx, 1.0);

  const double u_bottom = static_cast<double>(pts[0].x);
  f.u_err_near = (u_bottom - cx) / denom_u;
  f.u_err_lookahead = f.u_err_near;

  if (pts.size() >= 2) {
    double min_v = static_cast<double>(pts[0].y);
    double max_v = static_cast<double>(pts[0].y);
    double sum_v = 0.0;
    double sum_u = 0.0;
    for (const auto &p : pts) {
      const double v = static_cast<double>(p.y);
      const double u = static_cast<double>(p.x);
      min_v = std::min(min_v, v);
      max_v = std::max(max_v, v);
      sum_v += v;
      sum_u += u;
    }

    const double dv = max_v - min_v;
    if (dv > 1e-6) {
      const double inv_n = 1.0 / static_cast<double>(pts.size());
      const double mean_v = sum_v * inv_n;
      const double mean_u = sum_u * inv_n;

      double var_v = 0.0;
      double cov_vu = 0.0;
      for (const auto &p : pts) {
        const double v = static_cast<double>(p.y);
        const double u = static_cast<double>(p.x);
        var_v += (v - mean_v) * (v - mean_v);
        cov_vu += (v - mean_v) * (u - mean_u);
      }

      if (std::abs(var_v) > kEps) {
        const double a = cov_vu / var_v; // u = a*v + b
        const double b = mean_u - a * mean_v;
        f.slope = a / 120.0;

        double v_la = static_cast<double>(pts[0].y) - cfg.lookahead_delta_v_px;
        v_la = Clamp(v_la, min_v, max_v);
        const double u_la = a * v_la + b;
        f.u_err_lookahead = (u_la - cx) / denom_u;
      }
    }
  }

  bool in_recovery = previous_in_recovery;
  const bool enter = (f.n_visible <= cfg.recover_enter_nvis) ||
                     (std::abs(f.u_err_near) > cfg.recover_enter_u);
  const bool exit = (f.n_visible >= cfg.recover_exit_nvis) &&
                    (std::abs(f.u_err_near) < cfg.recover_exit_u);
  if (enter)
    in_recovery = true;
  if (exit)
    in_recovery = false;
  f.in_recovery = in_recovery ? 1.0 : 0.0;

  const bool enough_pts = f.n_visible >= 2.0;
  if (enough_pts) {
    const double alpha =
        in_recovery ? cfg.lookahead_alpha_recovery : cfg.lookahead_alpha_normal;
    const double a = Clamp(alpha, 0.0, 1.0);
    f.u_err_ctrl = (1.0 - a) * f.u_err_near + a * f.u_err_lookahead;
  } else {
    f.u_err_ctrl = f.u_err_near;
  }

  return f;
}

} // namespace vision
