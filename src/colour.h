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

#ifndef TONEMAP_COLOUR_H
#define TONEMAP_COLOUR_H

#include <cmath>
#include <string>

namespace tonemap {

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

constexpr Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out.r[i][j] =
                a.r[i][0] * b.r[0][j] + a.r[i][1] * b.r[1][j] + a.r[i][2] * b.r[2][j];
        }
    }
    return out;
}

// --- Primaries, BT.2100-3 Table 2 and BT.709 -------------------------------

struct Chromaticity {
    double x;
    double y;
};

struct Primaries {
    Chromaticity r;
    Chromaticity g;
    Chromaticity b;
};

inline constexpr Chromaticity kD65 = {0.3127, 0.3290};
inline constexpr Primaries kPrimariesBt2020 = {
    {0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}};
inline constexpr Primaries kPrimariesBt709 = {
    {0.640, 0.330}, {0.300, 0.600}, {0.150, 0.060}};
inline constexpr Primaries kPrimariesP3D65 = {
    {0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}};

// RGB to XYZ from primary and white chromaticities, by the BT.2087 method.
// P holds the primaries as (x, y, 1 - x - y) columns; solving P s = W for the
// white point as XYZ with Y = 1 gives the column scalings that put RGB
// (1, 1, 1) on the white point.
constexpr Mat3 rgbToXyz(const Primaries& p, const Chromaticity& w) {
    const Mat3 columns = {{
        {p.r.x, p.g.x, p.b.x},
        {p.r.y, p.g.y, p.b.y},
        {1.0 - p.r.x - p.r.y, 1.0 - p.g.x - p.g.y, 1.0 - p.b.x - p.b.y},
    }};
    const Mat3 back = inverse(columns);
    const double white[3] = {w.x / w.y, 1.0, (1.0 - w.x - w.y) / w.y};

    Mat3 out{};
    for (int j = 0; j < 3; ++j) {
        const double scale = back.r[j][0] * white[0] + back.r[j][1] * white[1] +
                             back.r[j][2] * white[2];
        for (int i = 0; i < 3; ++i) out.r[i][j] = columns.r[i][j] * scale;
    }
    return out;
}

inline constexpr Mat3 kRgb2020ToXyz = rgbToXyz(kPrimariesBt2020, kD65);
inline constexpr Mat3 kRgb709ToXyz = rgbToXyz(kPrimariesBt709, kD65);
inline constexpr Mat3 kXyzToRgb709 = inverse(kRgb709ToXyz);
inline constexpr Mat3 kRgb2020ToRgb709 = multiply(kXyzToRgb709, kRgb2020ToXyz);

// CIE 1976 u'v' of a chromaticity. u' = 4x / (-2x + 12y + 3) and
// v' = 9y / (-2x + 12y + 3), which is equation (5-3) of BT.2407 rewritten
// from XYZ into xy.
constexpr Chromaticity xyToUv(const Chromaticity& c) {
    const double d = -2.0 * c.x + 12.0 * c.y + 3.0;
    return {4.0 * c.x / d, 9.0 * c.y / d};
}

inline constexpr Chromaticity kWhiteUv = xyToUv(kD65);

inline constexpr Mat3 kRgb2020ToLms = {{
    {1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0},
    {683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0},
    {99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0},
}};
inline constexpr Mat3 kLmsToRgb2020 = inverse(kRgb2020ToLms);

// Annex 5 and BT.2100 Table 6 print these three luminance coefficients. They
// are signal coefficients rather than CIE luminance, so yrgb and ycbcr use
// them exactly as printed; the exact BT.2020 row differs by up to 2.9e-5
// relative on blue, which is above float32 tolerance. The gamut mapper, where
// Y really is luminance, uses the exact row instead.
inline constexpr double kKr = 0.2627;
inline constexpr double kKg = 0.6780;
inline constexpr double kKb = 0.0593;
inline constexpr double kCbDivisor = 2.0 * (1.0 - kKb);  // 1.8814
inline constexpr double kCrDivisor = 2.0 * (1.0 - kKr);  // 1.4746

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
// VapourSynth C boundary. minName and maxName are what the message calls the
// source pair, so a value that came from a frame property is reported under
// the property's name rather than under the argument it stood in for.
std::string makeEetf(double srcMin, double srcMax, double dstMin, double dstMax,
                     const char* minName, const char* maxName, Eetf* out);

// The checks whose inputs are all arguments, so they can run at create time
// and fail the script at evaluation rather than at the first frame.
std::string checkLuminance(const char* name, double value);
std::string checkTargetRange(double dstMin, double dstMax);

// min(v1/v2, v2/v1), taken as 1 when either value is zero.
template <typename T>
inline T chromaRatio(T v1, T v2) {
    if (v1 == T(0) || v2 == T(0)) return T(1);
    const T a = v1 / v2;
    const T b = v2 / v1;
    return a < b ? a : b;
}

}  // namespace tonemap

#endif  // TONEMAP_COLOUR_H
