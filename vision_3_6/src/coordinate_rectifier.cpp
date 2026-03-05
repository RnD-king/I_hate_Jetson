#include "vision/coordinate_rectifier.hpp"

#include <cmath>

namespace vision
{

CoordinateRectifier::CoordinateRectifier(const Intrinsics& K) // 라디안으로 받아오는 중
: K_(K)
{
}

std::vector<cv::Point2f> CoordinateRectifier::RectifyPixelPoints(
    const std::vector<cv::Point2f>& pts_px,
    double roll_rad,
    double pitch_rad) const
{
    std::vector<cv::Point2f> out;
    out.reserve(pts_px.size());

    // ---- 회전 행렬 구성 ----
    // 정의(위 헤더 주석과 동일):
    // roll  = Rz(roll)  : 카메라 전방축(z) 기준 회전
    // pitch = Rx(pitch) : 카메라 우측축(x) 기준 회전
    //
    // 카메라가 roll/pitch로 기울어져서 관측된 픽셀점을
    // "수평 카메라(roll=0,pitch=0)"에서 관측된 것처럼 만들려면
    // 광선을 역회전(inverse rotation) 시킨 뒤 재투영한다.
    //
    // R = Rz(roll) * Rx(pitch)
    // r_level = R^{-1} * r_cam = Rx(-pitch) * Rz(-roll) * r_cam

    const double cr = std::cos(-roll_rad);
    const double sr = std::sin(-roll_rad);
    const double cp = std::cos(-pitch_rad);
    const double sp = std::sin(-pitch_rad);

    // Rz(-roll)
    // [ cr -sr  0 ]
    // [ sr  cr  0 ]
    // [  0   0  1 ]

    // Rx(-pitch)
    // [ 1  0   0 ]
    // [ 0  cp -sp]
    // [ 0  sp  cp]

    const double eps = 1e-9;

    for (const auto& uv : pts_px){
        // 1) 픽셀 -> 정규화 좌표 -> 카메라 광선
        // (u - cx)/fx, (v - cy)/fy
        const double xn = (static_cast<double>(uv.x) - K_.cx) / (K_.fx + eps);
        const double yn = (static_cast<double>(uv.y) - K_.cy) / (K_.fy + eps);

        // 광선(카메라 좌표계)
        double x = xn;
        double y = yn;
        double z = 1.0;

        // 2) Rz(-roll) 적용
        // x1 = cr*x - sr*y
        // y1 = sr*x + cr*y
        // z1 = z
        const double x1 = cr * x - sr * y;
        const double y1 = sr * x + cr * y;
        const double z1 = z;

        // 3) Rx(-pitch) 적용   
        // x2 = x1
        // y2 = cp*y1 - sp*z1
        // z2 = sp*y1 + cp*z1
        const double x2 = x1;
        const double y2 = cp * y1 - sp * z1;
        const double z2 = sp * y1 + cp * z1;

        // 4) 재투영 (정규화 -> 픽셀)
        // z2가 0에 가까우면 수치 폭주하므로 원본을 그대로 쓰는 게 안전하다.
        if (std::abs(z2) < 1e-6){
            out.emplace_back(uv);
            continue;
        }

        const double u2 = K_.fx * (x2 / z2) + K_.cx;
        const double v2 = K_.fy * (y2 / z2) + K_.cy;

        out.emplace_back(static_cast<float>(u2), static_cast<float>(v2));
    } 
    return out;
}

}  // namespace vision