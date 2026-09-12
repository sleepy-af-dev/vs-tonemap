// PQ, the BT.2100 matrices and the BT.2408 Annex 5 EETF.
//
// Everything here is double. Section 5 of the design measures float32 PQ
// transforms at 3e-5 to 8e-5 relative error in linear output against a
// 1e-5 accuracy gate, because the outer exponent of 78.8 multiplies the
// rounding of the base and the decode multiplies that again by 5 to 10. The
// arithmetic type is a template parameter so the same source can be built
// with float lanes for the Phase 5 speed comparison.
//
// Sources: ITU-R BT.2100-3 Table 4 (PQ), Table 7 (ICtCp), ITU-R BT.2408-9
// Annex 5 (the EETF).

#ifndef TONEMAPPER_COLOUR_H
#define TONEMAPPER_COLOUR_H

#include <cmath>
#include <string>

namespace tonemapper {

// --- PQ, BT.2100-3 Table 4 -------------------------------------------------

inline constexpr double kM1 = 2610.0 / 16384.0;
inline constexpr double kM2 = 2523.0 / 4096.0 * 128.0;
inline constexpr double kC1 = 3424.0 / 4096.0;
inline constexpr double kC2 = 2413.0 / 4096.0 * 32.0;
inline constexpr double kC3 = 2392.0 / 4096.0 * 32.0;
inline constexpr double kPqPeak = 10000.0;  // cd/m2 at a PQ code of 1

// Absolute luminance in cd/m2 to a PQ code. The input is clamped to
// [0, 10000] because that is the whole domain PQ defines, so negative linear
// light from resampling ringing or sub-black codes reads as black.
template <typename T>
inline T pqInverseEotf(T luminance) {
    T y = luminance * static_cast<T>(1.0 / kPqPeak);
    y = y < T(0) ? T(0) : (y > T(1) ? T(1) : y);
    const T ym = std::pow(y, static_cast<T>(kM1));
    return std::pow((static_cast<T>(kC1) + static_cast<T>(kC2) * ym) /
                        (T(1) + static_cast<T>(kC3) * ym),
                    static_cast<T>(kM2));
}

// PQ code to absolute luminance in cd/m2. Codes below PQ(0), which is
// 7.31e-7 rather than 0, come back as black through the spec's own max(., 0).
template <typename T>
inline T pqEotf(T code) {
    const T t = std::pow(code < T(0) ? T(0) : code, static_cast<T>(1.0 / kM2));
    const T num = t - static_cast<T>(kC1);
    return static_cast<T>(kPqPeak) *
           std::pow((num < T(0) ? T(0) : num) /
                        (static_cast<T>(kC2) - static_cast<T>(kC3) * t),
                    static_cast<T>(1.0 / kM1));
}

// --- Matrices, BT.2100-3 Table 7 -------------------------------------------

struct Mat3 {
    double r[3][3];
};

// Inverses are computed here rather than copied from published decimals, so
// the round trip is exact to the last bit the type allows.
constexpr Mat3 inverse(const Mat3& a) {
    const double(&m)[3][3] = a.r;
    const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    return Mat3{{
        {c00 / det, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
         (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det},
        {c01 / det, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
         (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det},
        {c02 / det, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
         (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det},
    }};
}

template <typename T>
inline void applyMatrix(const Mat3& a, T x, T y, T z, T* out0, T* out1, T* out2) {
    const T v0 = static_cast<T>(a.r[0][0]) * x + static_cast<T>(a.r[0][1]) * y +
                 static_cast<T>(a.r[0][2]) * z;
    const T v1 = static_cast<T>(a.r[1][0]) * x + static_cast<T>(a.r[1][1]) * y +
                 static_cast<T>(a.r[1][2]) * z;
    const T v2 = static_cast<T>(a.r[2][0]) * x + static_cast<T>(a.r[2][1]) * y +
                 static_cast<T>(a.r[2][2]) * z;
    *out0 = v0;
    *out1 = v1;
    *out2 = v2;
}

inline constexpr Mat3 kRgb2020ToLms = {{
    {1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0},
    {683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0},
    {99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0},
}};
inline constexpr Mat3 kLmsToRgb2020 = inverse(kRgb2020ToLms);

inline constexpr Mat3 kLmspToIctcp = {{
    {2048.0 / 4096.0, 2048.0 / 4096.0, 0.0},
    {6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0},
    {17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0},
}};
inline constexpr Mat3 kIctcpToLmsp = inverse(kLmspToIctcp);

// --- The EETF, BT.2408-9 Annex 5 steps 1 to 5 ------------------------------

struct Eetf {
    double pqLb;
    double pqLw;
    double span;
    double minLum;
    double maxLum;
    double ks;
    bool degenerate;  // KS >= 1, so the spline is never evaluated

    template <typename T>
    T apply(T e) const {
        T e1 = (e - static_cast<T>(pqLb)) * static_cast<T>(1.0 / span);
        // The spec defines the curve on [0, 1] only, and the cubic is not
        // monotone outside it. Above the mastering peak reads as the peak,
        // below the mastering black as black.
        e1 = e1 < T(0) ? T(0) : (e1 > T(1) ? T(1) : e1);

        T e2 = e1;
        if (!degenerate && e1 >= static_cast<T>(ks)) {
            const T t = (e1 - static_cast<T>(ks)) / static_cast<T>(1.0 - ks);
            const T t2 = t * t;
            const T t3 = t2 * t;
            e2 = (T(2) * t3 - T(3) * t2 + T(1)) * static_cast<T>(ks) +
                 (t3 - T(2) * t2 + t) * static_cast<T>(1.0 - ks) +
                 (T(-2) * t3 + T(3) * t2) * static_cast<T>(maxLum);
        }

        const T lift = T(1) - e2;
        const T lift2 = lift * lift;
        const T e3 = e2 + static_cast<T>(minLum) * lift2 * lift2;
        return e3 * static_cast<T>(span) + static_cast<T>(pqLb);
    }

    // The luminance an exact-black input becomes, which is dst_min.
    double black() const { return pqEotf<double>(apply<double>(pqLb)); }
};

// Fills out and returns an empty string, or leaves it alone and returns the
// reason the four luminances cannot define a curve. No exceptions cross the
// VapourSynth C boundary.
std::string makeEetf(double srcMin, double srcMax, double dstMin, double dstMax,
                     Eetf* out);

// min(v1/v2, v2/v1), taken as 1 when either value is zero.
template <typename T>
inline T chromaRatio(T v1, T v2) {
    if (v1 == T(0) || v2 == T(0)) return T(1);
    const T a = v1 / v2;
    const T b = v2 / v1;
    return a < b ? a : b;
}

}  // namespace tonemapper

#endif  // TONEMAPPER_COLOUR_H
